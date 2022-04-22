import base64
import datetime
import io
import logging
import random
import time
import traceback
import uuid
import json
from json import JSONDecodeError
from collections import OrderedDict
from datetime import timedelta, date
from functools import lru_cache

from django.core.exceptions import ValidationError
from django.core.files import File
from django.db.models import Q, Prefetch, Max, Case, F, When, Sum
from raven.contrib.django.raven_compat.models import client as sentry
import pytz

from PyPDF2 import PdfFileWriter, PdfFileReader
from dateutil.relativedelta import relativedelta
from django.conf import settings
from django.contrib.gis.db import models as gis_models
from django.contrib.postgres.fields import JSONField, ArrayField
from rest_framework.exceptions import ValidationError
from django.core.files.base import ContentFile
from django.core.validators import RegexValidator
from django.db import models, transaction
from django.db.models import QuerySet, Subquery, OuterRef, Count, Q
from django.db.models.functions import Now, Least
from django.utils.functional import cached_property
from django.utils.translation import ugettext_lazy as _
from django_fsm import transition
from phonenumber_field.modelfields import PhoneNumberField
from social_django.models import USER_MODEL

from banking.choices import BankAccountType
from banking.choices.debit_instruction import DebitSponsorName
from banking.models import BaseBankAccount, BaseDebitInstruction, Bank, EmandateDestinationBanks
from base.choices import EntityStatus
from base.fsm import IntegerFSMField
from base.model_mixins import EntityMixin, EntityBaseManager, EntityBaseQuerySet
from base.models import (
    BaseModel,
    base_reversion_register,
    get_status_history_mcls,
    ReadMixin,
    EntityPublishedAtBaseModel,
)
from comms.choices.main import EmailTemplateName, SMSTemplateName, PushTemplateName
from core import utils
from core.compat import secrets
from core.models import TenantAwareModel
from core.restframework import BlockingValidationError
from core.utils import now
from core.utils.cache import LockAcquisitionFailed, robust_lock
from core.utils.db_connection import discrete_connection_handler
from docs.choices import DocumentType, SignatureType, DocumentRequirementCategory, DocumentSource
from docs.models import Document, UserInfoRequest
from docs.utils.document_groups import get_docs_for_category
from external.choices import Method, ExternalCallBackEventType
from external.models import PerfiosData, PerfiosTransaction
from geo.models import BaseAddress, PostalCode
from internal.models import ParserTransaction
from internal.services.ams import StatementParser, AssessmentServiceWrapper
from internal.services.payments import PaymentForeclosureDetails
from lms.choices.main import IIFLDocUploadStatus, IIFLDocsToUpload, LoanProvider
from lms.models import Loan
from partners.models import Partner, PartnerCompany
from products.choices import UnitType
from products.choices.main import EligibilityStatus, AMSProductType, LoanApplicationStatus
from products.models import LoanApplication
from products.models.loans import RejectionReason
from products.models.main import ProductEligibility, ProductEligibilityStatusHistory
from products.services import ProductPolicyProxy
from products.utils import PRODUCT_ELIGIBILITY_LOCK_KEY
from users.choices import (
    EmbeddedDataSource,
    PROVIDER_MAP,
    NachOrigin,
    NachRejectionAction,
    PerfiosSourceFormat,
    CKYCBaseSearchType,
    AXMLProceedOption,
    CKYCConsentType,
    PROVIDER_REVERSE_MAP,
)
from users.choices.employment import OfficialEmailVerificationStatus, EmployerVerificationMethod
from users.choices.main import (
    VehicleType,
    BankPassbookFraudStatus,
    BPDExternalStatus,
    OTPMethod,
    RBIMoratoriumEligibility,
    NameMismatchDeclarationStatus,
    NameMismatchDeclarationDocument,
    VerificationType,
    DataSource,
    AdditionalDataReferenceTable,
    AMLActionType,
)
from users.choices.report import ReportStatus
from users.choices.users import (
    UserLoanProvider,
    FDStatus,
    FullertonDocID,
    KYCVerificationMode,
    PayUFinLeadStatus,
    UserLoginAction,
    AuthUserBlockingStatus,
    BtTopUpPaymentOption,
)
from users.data import UNIT_QS_ANNOTATIONS
from users.utils.nach_resetup_comms_scheduler import NachReSetupCommsScheduler
from .choices import (
    Gender,
    EmailType,
    EmploymentType,
    PaymentMode,
    MasterUserStatus,
    OTPType,
    AddressType,
    AddressSurrogateType,
    NachStatus,
    ReportType,
    MaritalStatus,
    FullertonLeadStatus,
    EducationalQualification,
    SalariedEmployerType,
    SelfEmploymentCategory,
    Community,
    Category,
    MoratoriumMode,
    HouseOwner,
)
from .choices.external import SkipStep
from .constants import (
    CREATE_RESETUP_NACH_FOR_REMARK,
    NACH_TOTAL_MAX_ATTEMPT_REJECTION_REMARKS,
    MAX_NACH_REPROCESS_COUNT,
    MAX_NACH_TOTAL_ATTEMPTS,
    NACH_MAX_REPROCESS_REJECTION_REMARKS,
    CREATE_RESETUP_REMARKS,
    NACH_NO_ACTIVE_LOAN_REMARKS,
    NachResetupTypes,
    BYJU_VENDOR_CODE,
    NACH_REJECTED_BY_REJECTION_ACTION,
    NON_DIGIO_USER_LOAN_PROVIDER,
    DUMMY_EMAIL_DOMAIN,
    PLATFORM_LAZYPAY,
)
from .utils import generate_otp
from cpv.utils import CpvUtils
from .choices.nach import NachSubStatus
from banking.choices import DebitInstructionType
from core.models import CoreGlobalConfig
from docs.choices import DocumentStatus
from .utils.entity import create_and_link_nach

LOGGER = logging.getLogger(__name__)


class PhoneNumberQuerySet(QuerySet):
    def having_masteruser(self):
        return self.filter(masteruser__isnull=False)


@base_reversion_register()
class PhoneNumber(BaseModel):
    phone = PhoneNumberField(primary_key=True)
    is_dnd = models.BooleanField(default=False)
    otp_verified_at = models.DateTimeField(
        null=True,
        editable=False,
    )
    whatsapp_enabled = models.NullBooleanField()
    whatsapp_consent = models.NullBooleanField(
        help_text=_("Consent to message user using WhatsApp business number"),
    )
    objects = PhoneNumberQuerySet.as_manager()
    otp_skipped = models.BooleanField(default=False)

    def update_consent(self, whatsapp_consent=None):
        if not self.whatsapp_consent and whatsapp_consent is not None:
            self.whatsapp_consent = whatsapp_consent
            self.save()

    def reset_verification(self):
        self.otp_verified_at = None
        self.save()
        self.reset_otp_skipped()

    def reset_otp_skipped(self):
        self.otp_skipped = False
        self.save()

    def verify(self):
        if not self.otp_verified_at:
            self.otp_verified_at = utils.now()
            self.save()

    def remove_consent(self):
        self.whatsapp_consent = False
        self.save()

    def __str__(self):
        return str(self.phone)


class OTPQuerySet(QuerySet):
    def _annotate_validity(self):
        return self.annotate(
            valid_until=models.ExpressionWrapper(
                models.F("sent_at") + models.F("valid_for"), output_field=models.DateTimeField()
            )
        )

    def authorisation(self):
        return self.filter(otp_type=OTPType.Authorization)

    def authentication(self):
        return self.filter(otp_type=OTPType.Authentication)

    def valid(self):
        return self._annotate_validity().filter(valid_until__gte=Now()).order_by("-sent_at")


class OTP(BaseModel):
    """
    `sent_at` is to allow sending OTP asynchronously and only count
     sent OTPs against the limit.
    """

    phone = models.ForeignKey(PhoneNumber)
    code = models.CharField(max_length=8, default=generate_otp)
    otp_type = models.SmallIntegerField(
        choices=OTPType.choices,
        default=OTPType.Authentication,
    )
    sent_at = models.DateTimeField(editable=False, null=True)
    valid_for = models.DurationField(
        editable=False,
        default=timedelta(milliseconds=settings.OTP_VALIDITY_MS),
    )
    method = models.SmallIntegerField(choices=OTPMethod.choices, default=OTPMethod.SMS)

    objects = OTPQuerySet.as_manager()

    @classmethod
    def send(cls, phone, otp_type=OTPType.Authentication, method=OTPMethod.SMS):
        if OTP.is_demo_number(phone.phone):
            return cls.objects.create(phone=phone, otp_type=otp_type, method=method, code=OTP.get_demo_otp())

        return cls.objects.create(phone=phone, otp_type=otp_type, method=method)

    @classmethod
    def is_demo_number(cls, phone_id):
        return settings.DEMO_ENABLED and settings.DEMO_OTP_VALUES and phone_id in settings.DEMO_PHONE_NUMBERS

    @classmethod
    def get_demo_otp(cls):
        return random.choices(settings.DEMO_OTP_VALUES)[0]

    def save(self, *args, **kwargs):
        from .tasks import send_otp

        super().save(*args, **kwargs)
        if not self.sent_at:
            self.phone.reset_otp_skipped()

            try:
                trigger_partner_bureau_consent_template = self.trigger_partner_bureau_consent_template
            except (AttributeError, KeyError):
                trigger_partner_bureau_consent_template = False

            transaction.on_commit(lambda: send_otp.delay(self.pk, trigger_partner_bureau_consent_template))


class MoratoriumEligibilityHistory(models.Model, ReadMixin):
    moratorium = models.ForeignKey(
        "Moratorium",
        related_name="transitions",
        on_delete=models.CASCADE,
    )

    status = IntegerFSMField(
        choices=RBIMoratoriumEligibility.choices,
        default=None,
        null=True,
    )

    remarks = models.TextField(blank=True, null=True)
    origin = models.TextField()
    ts = models.DateTimeField(auto_now_add=True)

    @classmethod
    def store_transition(cls, moratorium, status, origin=None, remarks=None):
        obj = cls.objects.create(moratorium=moratorium, status=status, origin=origin, remarks=remarks)
        return obj

    def add_remarks(self, remarks):
        self.remarks = "|".join([self.remarks or "", remarks]).strip("|")


class Moratorium(BaseModel):
    HISTORY_STORE = MoratoriumEligibilityHistory
    master_user = models.ForeignKey("users.MasterUser")
    otp = models.ForeignKey(OTP, null=True, blank=True)
    moratorium_mode = models.SmallIntegerField(
        choices=MoratoriumMode.choices,
        default=MoratoriumMode.Explicit,
    )
    reason = models.CharField(max_length=255, blank=True, null=True)
    is_eligible = IntegerFSMField(
        choices=RBIMoratoriumEligibility.choices,
        default=None,
        null=True,
    )
    accepted_date = models.DateTimeField(null=True, blank=True)

    @transition(
        field=is_eligible,
        source=[RBIMoratoriumEligibility.Pending, RBIMoratoriumEligibility.Rejected],
        target=RBIMoratoriumEligibility.Approved,
    )
    def approve(self, origin=None, remarks=None):
        self.HISTORY_STORE.store_transition(
            self, RBIMoratoriumEligibility.Approved, origin=origin, remarks=remarks
        )
        self.accepted_date = utils.now()

    @transition(
        field=is_eligible,
        source=[RBIMoratoriumEligibility.Pending, RBIMoratoriumEligibility.Approved],
        target=RBIMoratoriumEligibility.Rejected,
    )
    def reject(self, origin=None, remarks=None):
        self.HISTORY_STORE.store_transition(
            self, RBIMoratoriumEligibility.Rejected, origin=origin, remarks=remarks
        )

    @classmethod
    def create(
        cls,
        user,
        reason,
        moratorium_mode=MoratoriumMode.Explicit,
        otp=None,
        origin=None,
        remarks=None,
        accepted_date=None,
        is_eligible=None,
    ):
        obj = cls.objects.create(
            master_user_id=user,
            otp=otp,
            moratorium_mode=moratorium_mode,
            reason=reason,
            accepted_date=accepted_date,
            is_eligible=is_eligible,
        )
        cls.HISTORY_STORE.store_transition(obj, is_eligible, origin=origin, remarks=remarks)
        return obj

    def save(self, *args, **kwargs):
        if self.is_eligible == RBIMoratoriumEligibility.Approved and not self.accepted_date:
            self.accepted_date = utils.now()
        return super().save(*args, **kwargs)

    @staticmethod
    def has_opted_for_moratorium(master_user_id):
        return Moratorium.objects.filter(master_user_id=master_user_id, accepted_date__isnull=False).exists()

    # class Meta:
    #     unique_together = (('master_user', 'accepted_date', 'is_eligible'),)


@base_reversion_register()
class Email(BaseModel):
    email = models.EmailField(primary_key=True)
    email_type = models.PositiveSmallIntegerField(
        choices=EmailType.choices,
        validators=[EmailType.validator],
        default=EmailType.Personal,
    )
    # if user is subscribed to promo/marketing emails
    promotional_subscribed = models.BooleanField(default=True)
    verified_at = models.DateTimeField(null=True, editable=False)

    # it returns true if email is a dummy email (domain is equal to 'DUMMY_EMAIL_DOMAIN')
    @property
    def is_dummy_email(self):
        return self.email.split("@")[1] == DUMMY_EMAIL_DOMAIN

    def reset_verification(self):
        self.verified_at = None
        self.save()

    def verify(self):
        if not self.verified_at:
            self.verified_at = utils.now()
            self.save()

    def __str__(self):
        return self.email


class SourceQuerySet(QuerySet):
    def external(self):
        return self.filter(external=True)


class Source(TenantAwareModel):
    key = models.CharField(max_length=60, unique=True)
    external = models.BooleanField(default=False)
    secret = models.CharField(
        max_length=32,
        null=True,
        editable=False,
        unique=True,
    )
    send_comms = models.BooleanField(default=False)
    skip_otp = models.BooleanField(default=False)
    skip_installation = models.BooleanField(default=False)
    platform_source = models.BooleanField(default=False)
    optimized_platform = models.BooleanField(default=False)
    assistance_required = models.BooleanField(default=False)
    landing_page_source = models.BooleanField(default=False)
    callback_url = models.CharField(max_length=120, null=True, blank=True)
    display_name = models.CharField(max_length=60, null=True, blank=True)
    nach_source = models.BooleanField(default=False)  # signifies if the partner does NACH at its end
    pre_submission_xml_required = models.BooleanField(default=False)
    validate_xml = models.BooleanField(default=True)
    skip_references = models.BooleanField(default=True)
    pre_submission_kyc_verification_required = models.BooleanField(default=False)
    can_auto_approve = models.BooleanField(default=False)
    minimum_req_contacts = models.IntegerField(
        default=10
    )  # denotes the minimum number of contacts/references
    # required from a user registered via this source
    allow_auto_verification = models.BooleanField(default=False)  # signifies if we can trust the partner for
    # docs/entity auto verification
    ten_lakh_eligibility = models.BooleanField(default=False)
    perfios_source_format = models.CharField(
        choices=PerfiosSourceFormat.choices, max_length=10, default=PerfiosSourceFormat.XML
    )
    partner_company = models.ForeignKey(PartnerCompany, null=True, editable=False)
    skip_cpv = models.BooleanField(default=False)
    skip_family_details = models.BooleanField(default=False)
    skip_image_match = models.BooleanField(default=False)
    match_reject_percentage = models.PositiveSmallIntegerField(blank=True, null=True)
    allow_optimized_user_registration = models.BooleanField(default=False)
    is_fldg_partner = models.BooleanField(default=False)
    allow_osa = models.BooleanField(default=False)
    enable_all_lenders = models.BooleanField(default=True)
    enable_lenders = ArrayField(
        models.IntegerField(choices=UserLoanProvider.choices), default=list(), blank=True, null=True
    )  # used when all the lenders are not enabled
    allow_flexible_tenure = models.BooleanField(default=False)
    # Whether loan plans should be fetched from AMS
    enable_ams_plans = models.BooleanField(default=True)
    skip_perfios = models.BooleanField(default=False)
    refinance_eligible = models.NullBooleanField()
    skip_mother_name = models.BooleanField(default=False)
    skip_penny_drop = models.BooleanField(default=False)
    contact_email_id = models.EmailField(blank=True, null=True)
    config = JSONField(
        blank=True, null=True
    )  # This field will, in the future, contain all the stuff written above
    objects = SourceQuerySet.as_manager()

    def is_lp(self):
        return self.key == PLATFORM_LAZYPAY

    def is_kredit_bee(self):
        return self.key == "kredit-bee"

    def is_go_upwards(self):
        return self.key == "go-upwards"

    def is_niro(self):
        return self.key == "niro"

    def clean(self):
        from users.utils.validators import SourceConfigValidator

        config = self.config
        if config:
            SourceConfigValidator(config)()

        if not self.pk and self.external is True and not self.secret:
            self.secret = secrets.token_urlsafe(24)

    def get_value(self, value):
        return self.config.get(value) if self.config else None

    def __str__(self):
        return self.key


class SourceRollout(BaseModel):
    iifl_rollout = models.FloatField(default=0.0)
    fullerton_rollout = models.FloatField(default=0.0)
    rbl_rollout = models.FloatField(default=0.0)
    northern_arc_rollout = models.FloatField(default=0.0)
    payufin_rollout = models.FloatField(default=0.0)
    ksf_rollout = models.FloatField(default=0.0)
    idfc_rollout = models.FloatField(default=0.0)
    source = models.ForeignKey(Source, null=True, blank=True)
    is_external = models.BooleanField(default=False)
    employment = models.CharField(choices=EmploymentType.choices, max_length=50, null=True, blank=True)

    def __str__(self):
        if self.is_external is False:
            return "%s (%s)" % ("Paysense", self.employment)
        elif self.source is None and self.is_external is True:
            return "%s (%s)" % ("Source Fallback", self.employment)
        else:
            return "%s (%s)" % (self.source, self.employment)

    def clean(self):
        if (
            SourceRollout.objects.exclude(id=self.id)
            .filter(employment=self.employment, source=self.source, is_external=self.is_external)
            .exists()
        ):
            raise ValidationError("Duplicate Entry")
        if (
            self.iifl_rollout
            + self.northern_arc_rollout
            + self.rbl_rollout
            + self.fullerton_rollout
            + self.payufin_rollout
            + self.ksf_rollout
            + self.idfc_rollout
            != 1
        ):
            raise ValidationError("addition of rollouts should be equal to 1")
        if self.source is not None and self.is_external is False:
            raise ValidationError("is_external should be enabled")
        if self.source is None and self.employment is None and self.is_external is False:
            raise ValidationError("Kindly add employment for this source")
        if (
            self.source
            and self.source.platform_source
            and (self.fullerton_rollout + self.payufin_rollout + self.ksf_rollout != 1)
        ):
            raise ValidationError("lenders available for platform source are fullerton/payufin/ksf")

    class Meta:
        unique_together = (("source", "employment"),)


def entity_completion_subquery(qs, application):
    filters = {}
    if application:
        filters["loan_application__id"] = application.id
    return Subquery(
        qs.filter(master_user=OuterRef("pk"), **filters).valid().values("is_complete")[:1],
        output_field=models.NullBooleanField(),
    )


def bpd_subquery():
    qs = BankPassbookData.objects.filter(master_user=OuterRef("pk"), is_valid=True)
    return Subquery(qs.values("pk")[:1])


def kyc_verification_subquery(application):
    filters = {"mode__isnull": False}
    if application:
        filters["loan_application__id"] = application.id
    qs = KYCVerification.objects.filter(master_user=OuterRef("pk"), is_valid=True, **filters)
    return Subquery(qs.values("pk")[:1])


def enach_subquery(application):
    qs = DebitInstruction.objects.filter(
        master_user=OuterRef("pk"),
        is_valid=True,
        instruction_type__in=[
            DebitInstructionType.ENACH_DIGIO,
            DebitInstructionType.ENACH_NPCI,
            DebitInstructionType.NACH_DIGITAL,
        ],
        loan_application__id=application.id,
    )
    return Subquery(qs.values("nach__status")[:1])


def aav_subquery(application):
    """
    This unit is complete when both address and kyc verification is done, if any one of them is pending then this unit
    is incomplete, we don't need to handle the condition if this unit is skippable we can handle it from can_skip_unit.
    Returns:
    """
    # Need to check if current address docs are done or current address is permanent
    return kyc_verification_subquery(application) and entity_completion_subquery(
        Address.objects.permanent(), application
    )


def house_ownership_existence_subquery():
    qs = HouseOwnership.objects.for_user_id(OuterRef("pk")).filter(is_valid=True)
    return Subquery(qs.values("pk")[:1])


class MasterUserQuerySet(QuerySet):

    TIMESTAMPS = {
        "credit_approved_ts": EligibilityStatus.Approved,
        "credit_declined_ts": EligibilityStatus.Declined,
        "credit_expired_ts": EligibilityStatus.Expired,
    }

    def annotate_entity_completion(self, application):
        annotations = {
            UNIT_QS_ANNOTATIONS[UnitType.Aadhaar]: entity_completion_subquery(
                Aadhaar.objects, None  # do not pass application as aadhaar is not linked to loan app
            ),
            UNIT_QS_ANNOTATIONS[UnitType.Employment]: entity_completion_subquery(
                Employment.objects, application
            ),
            UNIT_QS_ANNOTATIONS[UnitType.BankAccount]: entity_completion_subquery(
                BankAccount.objects, application
            ),
            UNIT_QS_ANNOTATIONS[UnitType.NACH]: entity_completion_subquery(
                DebitInstruction.objects, application
            ),
            UNIT_QS_ANNOTATIONS[UnitType.PreSalaryVerification]: bpd_subquery(),
            UNIT_QS_ANNOTATIONS[UnitType.KYCVerification]: kyc_verification_subquery(application),
            UNIT_QS_ANNOTATIONS[UnitType.NACHV3]: enach_subquery(application),
            UNIT_QS_ANNOTATIONS[UnitType.AadhaarAddress]: aav_subquery(application),
            UNIT_QS_ANNOTATIONS[UnitType.HouseOwnership]: house_ownership_existence_subquery(),
        }
        return self.annotate(**annotations)

    def with_timestamps(self, target_statuses=None):
        annotation_map = {
            timestamp_name: Max(
                Case(
                    When(
                        eligibilities__ams_product_type=AMSProductType.Flexi,
                        eligibilities__status=target_status,
                        then=F("eligibilities__transitions__ts"),
                    ),
                    default=None,
                )
            )
            for timestamp_name, target_status in self.TIMESTAMPS.items()
            if (target_statuses is None or target_status in target_statuses)
        }
        return self.annotate(**annotation_map)

    def with_latest_timestamps(self, target_statuses=None, ams_product_type=AMSProductType.Flexi):
        annotation_map = {
            timestamp_name: Max(
                Case(
                    When(
                        eligibilities__transitions__status=target_status,
                        eligibilities__ams_product_type=ams_product_type,
                        then=F("eligibilities__transitions__ts"),
                    ),
                    default=None,
                )
            )
            for timestamp_name, target_status in self.TIMESTAMPS.items()
            if (target_statuses is None or target_status in target_statuses)
        }
        return self.annotate(**annotation_map)

    def with_latest_assessed_at(self, ams_product_type=AMSProductType.Flexi):
        """
        last_assessed_at is the timestamp when user was assessed recently. This timestamp changes only when user is
        credit assessed for the first time or after the expiry. If the status changes from decline/approve to
        approve/decline/expire, this timestamp doesn't change.
        Args:
            ams_product_type:

        Returns: Queryset

        """
        return self.with_latest_timestamps(ams_product_type=ams_product_type).annotate(
            latest_assessed_at=Case(
                When(  # if statuses are in this order: declined, expired, approved
                    Q(credit_expired_ts__isnull=False, credit_approved_ts__gte=F("credit_expired_ts")),
                    then=F("credit_approved_ts"),
                ),
                When(  # if statuses are in this order: approved, expired, declined
                    Q(credit_expired_ts__isnull=False, credit_declined_ts__gte=F("credit_expired_ts")),
                    then=F("credit_declined_ts"),
                ),
                default=Least(F("credit_approved_ts"), F("credit_declined_ts")),
            )
        )


@base_reversion_register()
class MasterUser(BaseModel):
    FSM_FIELDS = ("status",)

    auth_user = models.OneToOneField(USER_MODEL, editable=False)
    status = IntegerFSMField(
        choices=MasterUserStatus.choices,
        default=MasterUserStatus.Registered,
        protected=True,
    )
    phone = models.OneToOneField(
        PhoneNumber,
        blank=True,
        null=True,
        on_delete=models.SET_NULL,
    )
    email = models.OneToOneField(
        Email,
        blank=True,
        null=True,
        on_delete=models.SET_NULL,
    )
    name = models.CharField(
        max_length=64,
        blank=True,
        null=True,
        help_text=_("What user needs to be addressed as"),
    )
    first_name = models.CharField(blank=True, null=True, max_length=64)
    middle_name = models.CharField(blank=True, null=True, max_length=64)
    last_name = models.CharField(blank=True, null=True, max_length=64)
    gender = models.CharField(choices=Gender.choices, max_length=2, blank=True, null=True)
    date_of_birth = models.DateField(blank=True, null=True)
    terms_accepted = models.BooleanField()
    legacy_product_type = models.CharField(max_length=12, null=True, editable=False)
    iifl_premium_eligible = models.NullBooleanField()
    top_up_intent = models.NullBooleanField()
    source = models.ForeignKey(Source, null=True, editable=False)
    partner = models.ForeignKey(Partner, null=True, editable=False, on_delete=models.SET_NULL)
    # coalesce(oldPsUserId, psUserId) from "boostUsers" table in old system
    # To be used for IIFL prospect creation and sharing ID externally
    customer_id = models.BigIntegerField(unique=True, editable=False)
    credit_expired_at = models.DateField(blank=True, null=True)
    # Since fullerton integration
    provider = models.PositiveSmallIntegerField(choices=UserLoanProvider.choices, null=True, blank=True)
    attribution_at = models.DateTimeField(blank=True, null=True)

    objects = MasterUserQuerySet.as_manager()

    def _do_insert(self, manager, using, fields, update_pk, raw):
        new_fields = [f for f in fields if f.name != "customer_id"]
        return super()._do_insert(manager, using, new_fields, update_pk, raw)

    def get_full_name(self):
        return " ".join(filter(None, [self.first_name, self.middle_name, self.last_name]))

    @property
    def whatsapp_phone_id(self):
        return "%s@c.us" % (self.phone_id.replace("+", ""),)

    @property
    def master_user_id(self):
        return self.pk

    @property
    def phone_verified(self):
        return self.phone.otp_verified_at is not None

    @property
    def email_verified(self):
        return self.email.verified_at is not None

    def should_prevent_native_credit_check(self):
        """
        Doing credit checks on native (Paysense) platforms should be prevented for users who've already been
        credit checked by certain other partners

        TODO: Revisit this for API lending if OTP verification is needed for them
        """
        if self.is_credit_approved() or self.is_credit_declined():
            source = self.source
            if source and source.get_value("prevent_native_credit_check"):
                return True
        return False

    @lru_cache(maxsize=8)
    def get_subseq_product_eligibility(self, installments=None, exclude_loan_id=None, fail_fast=False):
        """

        Args:
            installments:
            exclude_loan_id:
            fail_fast:

        Returns ProductEligibility:

        """
        from lms.data.product_eligiblity import SubsequentProductEligibility

        return SubsequentProductEligibility(
            self, installments=installments, exclude_loan_id=exclude_loan_id, fail_fast=fail_fast
        )

    @property
    def credit_check_required(self):
        eligibility = self.eligibilities.all().flexi().first()
        return not (
            eligibility
            and eligibility.status
            in (
                EligibilityStatus.Approved,
                EligibilityStatus.Declined,
            )
        )

    @property
    def overall_status(self):
        status_list = [EligibilityStatus.Approved, EligibilityStatus.Declined, EligibilityStatus.Expired]
        eligibility_status = self.eligibilities.filter(status__in=status_list).values_list(
            "status", flat=True
        )
        if EligibilityStatus.Approved in eligibility_status:
            return EligibilityStatus.Approved
        elif EligibilityStatus.Declined in eligibility_status:
            return EligibilityStatus.Declined
        elif EligibilityStatus.Expired in eligibility_status:
            return EligibilityStatus.Expired
        else:
            return 0

    def get_employment(self):
        # todo it returns user level employment. It should be removed when all apis are integrated with application id
        return Employment.objects.for_user(self).first()

    @cached_property
    def vendor_config(self):
        if self.partner and self.partner.vendor and hasattr(self.partner.vendor, "vendorconfig"):
            return self.partner.vendor.vendorconfig.config
        return None

    def _get_config(self, config_key, default_value=None):
        config = self.vendor_config
        if config:
            return config.get(config_key, default_value)

        return default_value

    @cached_property
    def should_send_comms(self):
        if self.source and not self.source.send_comms:
            return False
        return self._get_config("send_user_comms", True)

    @property
    def is_byju(self):
        if not self.partner or not self.partner.vendor:
            return False
        else:
            return self.partner.vendor.code == BYJU_VENDOR_CODE

    @property
    def has_partner(self):
        return self.partner_id is not None

    @property
    def has_source(self):
        return self.source_id is not None and self.partner_id is None

    @property
    def is_self_employed(self):
        employment = Employment.objects.for_user(self).first()
        if employment is None:
            return
        return employment.employment_type == EmploymentType.SelfEmployed

    @property
    def age(self):
        return relativedelta(date.today(), self.date_of_birth).years

    @property
    def can_trigger_callback_to_source(self):
        return self.source and self.source.callback_url

    def should_trigger_callback_to_source(self, callback_event):
        """
        This function uses a source level config to determine whether or not the callback API should be triggered
        for any particular event.

        The "config" column in the Source table is defined in the following manner (the integers represent
        ExternalCallbackEventType choices)
        {
            "callback_events": {
                "includes": [2, 3],
                "excludes": []
            }
        }
        The above value means the source wants the callbacks for just those 2 events and nothing else.

        Similarly, the following value means the source wants the callbacks for every event EXCEPT the 2 defined.
        {
            "callback_events": {
                "includes": [],
                "excludes": [2, 3]
            }
        }

        If a source wants EVERY single event, then there's no need to define anything and both "includes" and
        "excludes" can be left empty, or the entire value can be empty, such as
        {
            "callback_events": {
                "includes": [],
                "excludes": []
            }
        }
        {
            "callback_events": {}
        }

        If "callback_events" doesn't exist in the config JSON, that also is considered to include EVERY single event.

        Args:
            callback_event: <external.choices.external_callback.ExternalCallBackEventType> the event type

        Returns:
            <boolean> indicates whether the callback API should be triggered

        """
        if not self.can_trigger_callback_to_source:
            return False

        configured_callback_events = self.source.get_value("callback_events")
        if not configured_callback_events:
            return True

        includes = configured_callback_events.get("includes", [])
        excludes = configured_callback_events.get("excludes", [])
        if not includes and not excludes:
            return True

        if includes:
            return callback_event in includes

        if excludes:
            return not (callback_event in excludes)

    @cached_property
    def flexi_latest_assessed_at(self):
        return self.get_latest_assessed_at(AMSProductType.Flexi)

    def get_latest_assessed_at(self, ams_product_type=None):
        if ams_product_type is None:
            # Don't fetch applications before credit expiry
            # As attribution might have changed giving incorrect ams_product_type
            application = LoanApplication.objects.for_user(master_user_id=self.pk).first()
            if application:
                ams_product_type = application.ensure_ams_product_type()
        if ams_product_type is None:
            # ams_product_type can be none if no application exist or ams_product_type can't evaluated on application
            ams_product_type = AMSProductType.Flexi

        return (
            self._meta.model.objects.filter(id=self.pk)
            .with_latest_assessed_at(ams_product_type=ams_product_type)
            .first()
            .latest_assessed_at
        )

    def has_verified_aadhaar(self):
        from docs.data import get_user_uploaded_docs

        uploaded_docs = set(d["document_type"] for d in get_user_uploaded_docs(self.pk, verified_only=True))
        front_back = {DocumentType.AadhaarBack, DocumentType.AadhaarFront}
        front_back_uploaded = not (front_back - uploaded_docs)
        eaadhaar_uploaded = not ({DocumentType.EAadhaar} - uploaded_docs)
        if not front_back_uploaded and not eaadhaar_uploaded:
            return False

        aadhaar = Aadhaar.objects.for_user(self).first()
        return bool(aadhaar and aadhaar.number and aadhaar.is_approved())

    def has_fullerton(self):
        try:
            return self.fullerton is not None
        except FullertonExtra.DoesNotExist:
            return False

    def byju_decision_pending(self):
        return self.extra.byju_decision is None

    def get_latest_device_id(self):
        installation = self.installation_set.order_by("-last_opened_at").first()
        if installation is not None:
            return installation.device_id

    def get_source_name(self):
        if self.source and self.partner:
            return None
        elif self.source:
            return self.source.key
        elif self.partner:
            return self.partner.company
        else:
            return None

    def is_sourced_from_company(self, partner_company):
        return (self.has_source and self.source.partner_company_id == partner_company.id) or (
            self.has_partner and self.partner.company_mapping_id == partner_company.id
        )

    @classmethod
    def from_auth_user(cls, user):
        return cls(
            first_name=user.first_name,
            last_name=user.last_name,
            name=user.username,
            auth_user=user,
        )

    def _store_product_eligibility(self, user_assessment_data):

        for user_assessment in user_assessment_data:
            if user_assessment["is_approved"]:
                eligibility_status = EligibilityStatus.Approved
            else:
                eligibility_status = EligibilityStatus.Declined

            lock_key = PRODUCT_ELIGIBILITY_LOCK_KEY.format(master_user_id=self.pk)
            try:
                with robust_lock(lock_key):
                    LOGGER.info(
                        f"Product eligibility debug: in _store_product_eligibility, user_id:{self.pk} locked"
                    )
                    product_eligibility, created = ProductEligibility.objects.get_or_create(
                        master_user=self,
                        ams_product_type=user_assessment["ams_product_type"],
                        defaults={
                            "status": eligibility_status,
                            "assessment_lead_id": user_assessment["assessment_lead_id"],
                            "credit_line": user_assessment["credit_line"],
                        },
                    )
                    if created:
                        ProductEligibilityStatusHistory.objects.create(
                            product_eligibility=product_eligibility, status=eligibility_status
                        )

                        # First time eligibility -> Storing last assessed at field
                        product_eligibility.last_assessed_at = now()
                        product_eligibility.save()

                    else:
                        if user_assessment["is_approved"]:
                            product_eligibility.approve(
                                user_assessment["assessment_lead_id"], user_assessment["credit_line"]
                            )
                        else:
                            product_eligibility.decline(user_assessment["assessment_lead_id"])
                        # Storing last assessed at
                        product_eligibility.last_assessed_at = (
                            product_eligibility.master_user.get_latest_assessed_at(
                                ams_product_type=product_eligibility.ams_product_type
                            )
                        )
                        product_eligibility.save()
                LOGGER.info(
                    f"Product eligibility debug: in _store_product_eligibility, user_id:{self.pk} lock"
                    " released"
                )
            except LockAcquisitionFailed:
                sentry.captureException()
                LOGGER.error(f"Product eligibility debug: failed to acquire lock for user_id: {self.pk}")

    @utils.orm.ensure_atomic
    def credit_approve(self, user_assessment_data):
        self._store_product_eligibility(user_assessment_data=user_assessment_data)

    @utils.orm.ensure_atomic
    def credit_decline(self, user_assessment_data):
        self._store_product_eligibility(user_assessment_data=user_assessment_data)

        if self.should_trigger_callback_to_source(ExternalCallBackEventType.CreditDeclined):
            from external.tasks import trigger_external_callback

            transaction.on_commit(
                lambda: trigger_external_callback.delay(
                    self.pk, ExternalCallBackEventType.CreditDeclined, self.source_id, int(time.time())
                )
            )

    @DeprecationWarning
    def expire_user_features(
        self,
        expire_all=True,
        expire_docs=True,
        expire_perfios=True,
        expire_nmi=True,
        expire_bankpassbookdata=True,
    ):
        pans = PAN.objects.for_user(self)
        employments = Employment.objects.for_user(self)
        addresses = Address.objects.for_user(self)
        aadhaars = Aadhaar.objects.for_user(self)
        banks = BankAccount.objects.for_user(self)
        dis = DebitInstruction.objects.for_user(self)
        kyc = KYCVerification.objects.for_user(self)
        ckyc = CKYC.objects.for_user(self)
        refi_foreclosure = RefinanceForeclosureDetails.objects.for_user(self)

        reset_to_pending_entities = []
        if expire_all:
            entity_query_sets = [banks, dis, pans, aadhaars, employments, addresses, kyc, ckyc]
        else:
            entity_query_sets = [aadhaars, banks, dis, addresses.exclude(address_type=AddressType.Current)]
            reset_to_pending_entities = [pans, employments, addresses.current()]

        for query_set in reset_to_pending_entities:
            for e in query_set:
                e.reset_to_pending()
                e.save()

        for query_set in entity_query_sets:
            for e in query_set:
                e.expire()
                e.save()

        if expire_docs:
            docs = Document.objects.for_user(self.id)
            for doc in docs:
                doc.expire()
                doc.save()

        if expire_perfios:
            pds = PerfiosData.objects.for_user(self)
            for pd in pds:
                pd.invalidate()

            StatementParser.expire_statement_data([self.id])
            parser_txns = ParserTransaction.objects.for_user(self.id)
            for parser_txn in parser_txns:
                parser_txn.expire()

        if expire_nmi:
            for info_request in UserInfoRequest.objects.for_user(self.id):
                info_request.close()
                info_request.save()

        if expire_bankpassbookdata:
            bpds = BankPassbookData.objects.valid(self.id)
            for bpd in bpds:
                bpd.is_valid = False
                bpd.save(update_fields=["is_valid"])

        # expire refinance foreclosure details and related tradelines
        for foreclosure in refi_foreclosure:
            foreclosure.invalidate()
            foreclosure.save()

    def handle_user_features(self, expire_perfios=True, expire_nmi=True, expire_bankpassbookdata=True):
        """
        In order to make the returning journey for credit expired users an easier experience:
        1. entities would not be expired, instead they would be reset to pending state
        2. documents would not be expired, instead they would be moved to created state
        """
        from external.services.perfiosFCU.utils import Utils

        # expire owned house address if it exists for a user as this is an optional field which user may or may not
        # choose to enter in their returning journey
        owned_house_addresses = Address.objects.for_user(self).owned()
        entities_to_expire = [owned_house_addresses]
        for entities in entities_to_expire:
            for entity in entities:
                entity.expire()
                entity.save()

        reset_to_pending_entities = [PAN, Employment, Address, Aadhaar, BankAccount, DebitInstruction, CKYC]
        for entity in reset_to_pending_entities:
            query_set = entity.objects.for_user(self)
            for e in query_set:
                e.reset_to_pending()
                e.save()

        docs = Document.objects.for_user(self.id)
        for doc in docs:
            doc.move_to_created()
            doc.save()

        if expire_perfios:
            pds = PerfiosData.objects.for_user(self)
            for pd in pds:
                pd.invalidate()

            StatementParser.expire_statement_data([self.id])
            parser_txns = ParserTransaction.objects.for_user(self.id)
            for parser_txn in parser_txns:
                parser_txn.expire()

        if expire_nmi:
            for info_request in UserInfoRequest.objects.for_user(self.id):
                info_request.close()
                info_request.save()

        if expire_bankpassbookdata:
            bpds = BankPassbookData.objects.valid(self.id)
            for bpd in bpds:
                bpd.is_valid = False
                bpd.save(update_fields=["is_valid"])

        # expire refinance foreclosure details and related tradelines
        refi_foreclosure = RefinanceForeclosureDetails.objects.for_user(self)
        for foreclosure in refi_foreclosure:
            foreclosure.invalidate()
            foreclosure.save()

    @utils.orm.ensure_atomic
    def credit_expire(self, ams_product_type=AMSProductType.Flexi, assessment_lead_id=None):
        from comms.tasks import send_communication

        # expiring all available eligibilities
        eligibilities = self.eligibilities.filter(
            status__in=[EligibilityStatus.Approved, EligibilityStatus.Declined]
        )
        for eligibility in eligibilities:
            eligibility.expire()
            eligibility.save()

        # TODO: the if below shouldn't actually evaluate to True ideally, but I'm keeping it as it is to cover for users
        #  who don't have any record in ProductEligibility. However, this should be deprecated later after a sufficient
        #  length of time (when every user has gotten their record created in the table)
        if not eligibilities:
            try:
                lock_key = PRODUCT_ELIGIBILITY_LOCK_KEY.format(master_user_id=self.pk)
                with robust_lock(lock_key):
                    LOGGER.info(f"Product eligibility debug: in credit_expire, user_id:{self.pk} locked")
                    product_eligibility, created = ProductEligibility.objects.get_or_create(
                        master_user=self,
                        ams_product_type=ams_product_type,
                        defaults={
                            "status": EligibilityStatus.Expired,
                            "assessment_lead_id": assessment_lead_id,
                        },
                    )
                    if created:
                        ProductEligibilityStatusHistory.objects.create(
                            product_eligibility=product_eligibility, status=EligibilityStatus.Expired
                        )
                    else:
                        product_eligibility.expire()
                        product_eligibility.save()
                LOGGER.info(f"Product eligibility debug: in credit_expire, user_id:{self.pk} lock released")
            except LockAcquisitionFailed:
                sentry.captureException()
                LOGGER.error(
                    "Product eligibility debug: in credit_expire failed to acquire lock for user_id:"
                    f" {self.pk}"
                )

        # to send comms in the user friendly time
        if self.should_send_comms and not self.is_credit_declined():
            send_communication.delay(
                [self.id],
                email_template_name=EmailTemplateName.CreditExpired,
                sms_template_name=SMSTemplateName.CreditExpired,
                push_template_name=PushTemplateName.get_provider_template(
                    PushTemplateName.CreditExpired,
                    self.latest_application.provider if self.latest_application else None,
                ),
                timestamp=int(utils.timestamp_from_datetime(utils.now() + datetime.timedelta(hours=6))),
            )

        self.credit_expired_at = utils.now()
        self.phone.reset_verification()

        LOGGER.info(
            "Credit expired at after assigning value: {} for user id: {}".format(
                self.credit_expired_at, self.id
            )
        )
        self.partner_id = None

        should_trigger_callback_to_source = self.should_trigger_callback_to_source(
            ExternalCallBackEventType.CreditExpired
        )
        if self.source_id:
            _ = UserSourceChangeLog.objects.create(
                master_user=self, source_id=self.source_id, status=self.status
            )
            old_source_id = self.source_id
            self.source_id = None
        else:
            old_source_id = None

        self.attribution_at = None

        # mark CPV as credit expired
        CpvUtils.cpv_user_expire(master_user_id=self.master_user_id)

        # Invalidate IDFCMetaData entry for the user
        IDFCMetaData.invalidate(self.master_user_id)

        # Invalidate the house ownership details for the user, if it exists
        owned_house = self.owned_house.valid().first()
        if owned_house:
            owned_house.invalidate()
            owned_house.save()

        if should_trigger_callback_to_source:
            from external.tasks import trigger_external_callback

            trigger_external_callback.delay(
                self.pk, ExternalCallBackEventType.CreditExpired, old_source_id, int(time.time())
            )

    def credit_assessed(
        self, assessment, source=None, is_experian_cbp_pulled=False, refi_assessment=None, product_code=None
    ):
        from comms.tasks import send_communication
        from .tasks import send_whatsapp_message_credit_approved

        product_policy = ProductPolicyProxy(product_code)
        refi_is_approved = False
        refi_credit_line = 0
        if refi_assessment:
            refi_is_approved = refi_assessment.is_approved
            refi_credit_line = refi_assessment.credit_line
        is_approved = assessment.is_approved or refi_is_approved
        credit_line = (
            (assessment.credit_line or refi_credit_line) if product_policy.credit_line_available() else None
        )
        is_cibil_cbp_pulled = assessment.is_cibil_cbp_pulled

        user_assessment_data = [
            {
                "ams_product_type": assessment.product_type,
                "assessment_lead_id": assessment.assessment_lead_id,
                "is_approved": assessment.is_approved,
                "credit_line": credit_line,
            }
        ]

        if refi_assessment:
            user_assessment_data.append(
                {
                    "ams_product_type": AMSProductType.Refinance,
                    "assessment_lead_id": refi_assessment.assessment_lead_id,
                    "is_approved": refi_assessment.is_approved,
                    "credit_line": refi_assessment.credit_line,
                }
            )

        LOGGER.info(
            f"Debugging latency for master user: {self.id} logs from credit_assessed transaction start"
        )
        if is_approved is True:
            setattr(self, "_credit_line", credit_line)
            self.credit_approve(user_assessment_data)
        else:
            LOGGER.info(f"Debugging latency for master user: {self.id} logs from credit_assessed start")
            self.credit_decline(user_assessment_data)

        self.save()
        LOGGER.info(f"Debugging latency for master user: {self.id} logs from credit_assessed end")

        if self.should_send_comms and (not source or source.send_comms):
            if is_approved is True:
                if self.source_id and self.source.send_comms:
                    email_template = EmailTemplateName.CreditApprovedReminderExternalUser
                    sms_template = SMSTemplateName.CreditApprovedReminderExternalUser
                else:
                    email_template = EmailTemplateName.CreditApproved
                    sms_template = SMSTemplateName.CreditApproved

                send_communication.delay(
                    [self.master_user_id],
                    email_template_name=email_template,
                    sms_template_name=sms_template,
                    data={
                        "credit_limit": float(credit_line),
                        # TODO: remove dependency in for provider in comms template
                        # 'provider': PROVIDER_MAP[self.provider]
                    },
                    timestamp=int(utils.timestamp_from_datetime(utils.now() + timedelta(minutes=5))),
                )
                send_whatsapp_message_credit_approved.apply_async(
                    args=(self.pk, self.first_name, credit_line), countdown=150
                )
                # sending bt credit approved communication
                if refi_is_approved and refi_credit_line:
                    send_communication.delay(
                        [self.master_user_id],
                        sms_template_name=SMSTemplateName.CreditApprovedBtUser,
                        data={"credit_line": float(refi_credit_line), "first_name": self.first_name},
                    )
            else:
                if source:
                    if source.display_name:
                        data = {"partner": source.display_name}
                    else:
                        data = None
                else:
                    data = {"partner": ""}
                send_communication.delay(
                    [self.master_user_id],
                    email_template_name=EmailTemplateName.CreditDeclined,
                    sms_template_name=SMSTemplateName.CreditRejected,
                    data=data,
                )

        if is_approved is True and self.is_byju and not settings.DEBUG:
            application = self.latest_application
            lender = application.provider.upper() if application else None
            send_communication.delay(
                destinations={"email": settings.BYJU_OPS_ALERT_EMAIL},
                email_template_name=EmailTemplateName.ByjuCreditApproved,
                data={"master_user_id": self.id, "source": BYJU_VENDOR_CODE, "lender": lender},
            )

        if is_cibil_cbp_pulled is True or is_experian_cbp_pulled is True:

            if is_cibil_cbp_pulled is True and is_experian_cbp_pulled is True:
                sms_template = SMSTemplateName.CibilExperianCbpSms

            elif is_cibil_cbp_pulled is True:
                sms_template = SMSTemplateName.CibilCbpSms

            elif is_experian_cbp_pulled is True:
                sms_template = SMSTemplateName.ExperianCbpSms

            send_communication.delay(
                [self.master_user_id],
                sms_template_name=sms_template,
                data={
                    "master_user_id": self.id,
                    "experian_credit_report": settings.CONSUMER_PULL_REPORT_URL,
                    "cibil_credit_report": settings.CIBIL_CONSUMER_PULL_REPORT_URL,
                },
                timestamp=int(utils.timestamp_from_datetime(utils.now() + timedelta(minutes=5))),
            )

    @transaction.atomic()
    def associate_partner(self, partner_id, update_on_change=False):
        """
        Update partner id if it doesn't exist, change only on CreditCheck
        :param partner_id:
        :param update_on_change:
        :return:
        """
        if not self.partner_id or update_on_change:
            self.partner_id = partner_id
            if self.source_id is not None:
                UserSourceChangeLog.objects.create(
                    master_user=self, source_id=self.source_id, status=self.status
                )
                self.source_id = None
            self.attribution_at = utils.now()
            self.save()

    @transaction.atomic()
    def reset_source_partner(self):
        if self.source_id is not None:
            UserSourceChangeLog.objects.create(
                master_user=self, source_id=self.source_id, status=self.status
            )
            self.source_id = None

        if self.partner_id is not None:
            self.partner_id = None

        self.attribution_at = utils.now()
        self.save()

    @transaction.atomic()
    def update_source_and_partner(self, utm_source):
        try:
            source = Source.objects.get(key__iexact=utm_source)

            if source.id != self.source_id:
                if self.source_id is not None:
                    UserSourceChangeLog.objects.create(
                        master_user=self, source_id=self.source_id, status=self.status
                    )

                if self.partner_id is not None:
                    self.partner_id = None

                self.source_id = source.id

            self.attribution_at = utils.now()
            self.save()

        except Source.DoesNotExist:
            self.reset_source_partner()

    def should_assess_credit(self, ams_product_type=None):
        return self.is_registered(ams_product_type=ams_product_type) or self.is_credit_expired(
            ams_product_type=ams_product_type
        )

    def get_provider_eligibility_mapping(self):
        try:
            from lms.utils.lender import Lender, LenderRoutingException

            return Lender.get_lender_check_eligibility()
        except LenderRoutingException as e:
            raise BlockingValidationError(str(e))

    def is_registered(self, ams_product_type=None):
        if ams_product_type:
            return not self.eligibilities.filter(ams_product_type=ams_product_type).exists()

        ams_product_type = AMSProductType.Flexi
        if self.latest_application and self.latest_application.ensure_ams_product_type(recompute=True):
            ams_product_type = self.latest_application.ensure_ams_product_type(recompute=True)
        return not self.eligibilities.filter(ams_product_type=ams_product_type).exists()

    @cached_property
    def latest_application(self):
        return LoanApplication.objects.filter(master_user_id=self.id).order_by("-created_at").first()

    def is_credit_declined(self, ams_product_type=None):
        if ams_product_type:
            return self.eligibilities.all().declined(ams_product_type=ams_product_type).exists()

        return (
            self.eligibilities.all().exists()
            and not self.eligibilities.filter(
                status__in=[EligibilityStatus.Approved, EligibilityStatus.Expired]
            ).exists()
        )

    def is_credit_expired(self, ams_product_type=None):
        if ams_product_type:
            return self.eligibilities.all().expired(ams_product_type=ams_product_type).exists()

        return (
            self.eligibilities.all().exists()
            and self.eligibilities.filter(status=EligibilityStatus.Expired).count()
            == self.eligibilities.count()
        )

    def is_credit_approved(self, ams_product_type=None):
        if ams_product_type:
            return self.eligibilities.all().approved(ams_product_type=ams_product_type).exists()
        return self.eligibilities.filter(status=EligibilityStatus.Approved).exists()

    def __str__(self):
        return "{} ({})".format(self.phone, self.email)

    def basic_profile(self):
        return {
            "first_name": self.first_name,
            "full_name": self.get_full_name(),
            "name": self.name,
            "email": self.email_id,
            "phone": str(self.phone),
        }

    @cached_property
    def is_nach_active(self):
        debit_instruction = self.debitinstruction_set.valid().first()
        if not debit_instruction:
            return False
        return debit_instruction.nach.status == NachStatus.Active

    @staticmethod
    def apply_pf_function():
        if settings.APPLY_PF:
            return random.random() < settings.PF_ROLLOUT
        return None

    def apply_pf(self):
        try:
            master_user_extra = self.extra
        except MasterUserExtra.DoesNotExist:
            master_user_extra = MasterUserExtra.objects.create(
                master_user=self, apply_pf=self.apply_pf_function()
            )
        if master_user_extra.apply_pf is None:
            master_user_extra.apply_pf = self.apply_pf_function()
            master_user_extra.save()
        return master_user_extra.apply_pf or False

    def existing_loan_foreclosure_details(self):
        """
        foreclosure details of all the active loans of the user.

            Returns: PaymentForeclosureDetails

        """
        loans = Loan.objects.filter(master_user_id=self.id).active_loans()
        if not loans:
            return Loan.objects.none()

        loan_provider_loan_mapping = {loan.id: loan.provider_loan_id for loan in loans}

        return PaymentForeclosureDetails.get(
            list(loan_provider_loan_mapping.keys()), loan_provider_loan_mapping=loan_provider_loan_mapping
        )

    def salary_verification_attempted(self, application_id):
        """
        To be used only for non passbook related salary verification
        """
        application = LoanApplication.objects.get(pk=application_id)
        return (
            application.perfios.valid().filter(master_user_id=self.id).exists()
            or ParserTransaction.objects.filter(master_user_id=self.id).exists()
        )

    def salary_verification_method(self, since=None):
        """
        :return: Return None if salary verification is not found
        To be only used for non bank passbook verification methods
        """
        perfios_data = PerfiosData.objects.for_user(self, since).first()
        parser_txn = ParserTransaction.objects.successful(self.id, since).first()

        if perfios_data or parser_txn:
            if perfios_data and (
                not parser_txn or perfios_data.transaction.created_at > parser_txn.created_at
            ):
                return perfios_data.transaction.method
            return Method.Statement

    def get_external_registration_details(self):
        return self.externalregistration_set.filter(source_id=self.source_id).order_by("-created_at").first()

    def is_refinance_eligible(self, source=None):
        """
        master users can have the linked company mapping (eligible for refinance)
        either through partner or source (both used for users coming through partners)
        Also, source is not assigned initially when calling from External credit check APIs
        """
        return (
            self.is_company_refinance_eligible(source=source)
            and self.eligibilities.filter(
                ams_product_type=AMSProductType.Refinance, status=EligibilityStatus.Approved
            ).exists()
        )

    def is_company_refinance_eligible(self, source=None):
        source = source if source else self.source
        return (
            self.partner_id
            and self.partner.company_mapping_id
            and self.partner.company_mapping.refinance_eligible
        ) or (source and source.partner_company_id and source.partner_company.refinance_eligible)

    def get_ams_product_type(self, exclude_loan_id=None, product_eligibility=None):
        if self.loanapplication_set.disbursed().exists():
            if not product_eligibility:
                product_eligibility = self.get_subseq_product_eligibility(exclude_loan_id=exclude_loan_id)
            return product_eligibility.ams_product
        else:
            latest_app = LoanApplication.objects.for_user_v2(self.id).first()
            if latest_app:
                return latest_app.ensure_ams_product_type()
            return AMSProductType.Flexi

    class Meta:
        permissions = (
            ("read_masteruser", "Can read master user"),
            ("read_anymasteruser", "Can read master user across tenant/partner-company boundaries"),
            ("reassess_masteruser", "Can reassess masteruser"),
            ("post_assess_masteruser", "Can trigger post assess"),
            ("nach_write_master_user", "Can manage nach apis"),
            ("set_top_up_intent", "Can set top up intent"),
            ("read_sms", "Can view sms of master user"),
        )
        indexes = [
            models.Index(fields=["updated_at"]),
        ]


class AadhaarHistory(metaclass=get_status_history_mcls("AadhaarHistory", "users.Aadhaar")):
    pass


@base_reversion_register(exclude=("embedded_data",))
class Aadhaar(EntityMixin, BaseModel):
    HISTORY_STORE = AadhaarHistory
    number = models.CharField(
        validators=[
            RegexValidator(
                regex="\d{12}",
                message="Aadhaar number must be of 12 digits",
                code="invalid_aadhaar",
            )
        ],
        max_length=14,
        blank=True,
        null=True,
    )
    vid = models.CharField(
        validators=[
            RegexValidator(
                regex="\d{16}",
                message="VID must be of 16 digits",
                code="invalid_vid",
            )
        ],
        max_length=32,
        blank=True,
        null=True,
    )
    name = models.CharField(blank=True, null=True, max_length=255)
    co = models.CharField(blank=True, null=True, max_length=255)
    gender = models.CharField(
        choices=Gender.choices,
        validators=[Gender.validator],
        max_length=2,
        blank=True,
        null=True,
    )
    # Add validation for year_of_birth
    date_of_birth = models.DateField(blank=True, null=True)
    year_of_birth = models.CharField(
        max_length=4,
        null=True,
        blank=True,
    )
    phone_linked = models.NullBooleanField()
    embedded_data = JSONField(
        verbose_name=_("QR Data"),
        blank=True,
        null=True,
        help_text=_("QR code data on Aadhaar Card"),
    )
    embedded_data_source = models.PositiveSmallIntegerField(
        choices=EmbeddedDataSource.choices,
        blank=True,
        null=True,
    )
    manual_data = JSONField(
        blank=True,
        null=True,
        help_text=_("Aadhaar data entered manually by the user"),
    )
    address = models.ForeignKey(
        "Address",
        editable=False,
        null=True,
        on_delete=models.SET_NULL,
    )
    master_user = models.ForeignKey(MasterUser)
    eaadhaar = models.ForeignKey(
        "docs.Document",
        blank=True,
        null=True,
        on_delete=models.SET_NULL,
    )
    raw_data = JSONField(
        blank=True,
        null=True,
    )

    @property
    def address_is_current(self):
        return bool(self.address and self.address.is_current())

    def has_eaadhaar_data(self):
        return (
            self.eaadhaar_id is not None
            and self.embedded_data is not None
            and self.embedded_data_source == EmbeddedDataSource.EAadhaar
        )

    def has_qr_data(self):
        return self.embedded_data_source == EmbeddedDataSource.QR and self.embedded_data is not None

    @property
    def skip_qr(self):
        return not bool(self.embedded_data)

    def can_complete(self):
        if getattr(self, "_override", False) is True:
            return True

        has_aadhaar_address = getattr(self, "address", None) is not None
        application = LoanApplication.objects.for_user_v2(self.master_user_id).first()
        current_address = application.addresses.valid().current().first()
        has_current_address = current_address and not current_address.has_default_line1()

        if self.has_eaadhaar_data():
            return bool(has_aadhaar_address and has_current_address)
        else:
            return bool(has_aadhaar_address or has_current_address)

    def can_submit(self):
        return self.can_complete()

    def can_reject(self):
        return bool(self.pk)

    @transition(
        "status",
        source=(
            EntityStatus.Submitted,
            EntityStatus.Completed,
            EntityStatus.PreApproved,
            EntityStatus.Pending,
        ),
        target=EntityStatus.Rejected,
        conditions=[can_reject],
    )
    def reject(self, auth_user=None):
        kyc_obj = (
            Aadhaar.objects.prefetch_related(
                Prefetch("kycverification_set", queryset=KYCVerification.objects.filter(is_valid=True))
            )
            .get(pk=self.pk)
            .kycverification_set.first()
        )

        if kyc_obj is not None:
            kyc_obj.add_remarks("rejected by ops")
            kyc_obj.invalidate()

        if self.address:
            self.address.reject(auth_user=auth_user)
            self.address.save()

    def can_pre_approve(self):
        return self.can_submit()

    @transition(
        "status",
        source=(EntityStatus.Pending, EntityStatus.Completed, EntityStatus.Submitted),
        target=EntityStatus.Submitted,
        conditions=[can_submit],
    )
    def submit(self, auth_user=None):
        if self.address and not self.address.is_submitted():
            self.address.submit(auth_user=auth_user)
            self.address.save()

    @transition(
        "status",
        source=(EntityStatus.Submitted, EntityStatus.Completed, EntityStatus.PreApproved),
        target=EntityStatus.PreApproved,
        conditions=[can_pre_approve],
    )
    def pre_approve(self, auth_user=None):
        if self.address:
            self.address.pre_approve(auth_user=auth_user)
            self.address.save()

    @transition(
        "status",
        source=(EntityStatus.PreApproved, EntityStatus.Approved),
        target=EntityStatus.Approved,
        conditions=[can_pre_approve],
    )
    def approve(self, auth_user=None):
        if self.address:
            self.address.approve(auth_user=auth_user)
            self.address.save()

    def __str__(self):
        return str(self.number)


class PANHistory(metaclass=get_status_history_mcls("PANHistory", "users.PAN")):
    pass


class ExternalAdditionalDataMixin(models.Model):
    external_additional_data = models.ForeignKey("users.ExternalAdditionalData", blank=True, null=True)
    # To add verified by later when auto verifying
    # verified_by = models.PositiveSmallIntegerField(
    #     choices=VerifiedBy.choices,
    #     validators=[VerifiedBy.validator],
    #     blank=True,
    #     null=True
    # )

    def save_external_additional_data(self, external_additional_data, reference_table):
        if not external_additional_data:
            self.external_additional_data = None
        else:
            external_additional_data["reference_table"] = reference_table
            external_additional_data["reference_table_id"] = self.pk
            self.external_additional_data = ExternalAdditionalData.objects.create(**external_additional_data)

    class Meta:
        abstract = True


@base_reversion_register()
class PAN(EntityMixin, BaseModel, ExternalAdditionalDataMixin):
    HISTORY_STORE = PANHistory
    number = models.CharField(max_length=12)
    gender = models.CharField(
        choices=Gender.choices,
        validators=[Gender.validator],
        max_length=2,
        blank=True,
    )
    date_of_birth = models.DateField(blank=True, null=True)
    # if nsdl_name split (first/middle/last) is available, below fields are set
    first_name = models.CharField(max_length=128, editable=False, null=True)
    middle_name = models.CharField(max_length=128, editable=False, null=True)
    last_name = models.CharField(max_length=128, editable=False, null=True)
    # nsdl_name should always be populated with full_name
    nsdl_name = models.CharField(max_length=255, editable=False, blank=True)
    nsdl_valid = models.NullBooleanField(editable=False)
    master_user = models.ForeignKey(MasterUser)
    loan_application = models.ManyToManyField(LoanApplication, blank=True, related_name="pans")

    def can_complete(self):
        return bool(self.number)

    def can_submit(self):
        return self.can_complete()

    @classmethod
    def update_or_create_pending(cls, master_user, data):
        return cls.objects.update_or_create(
            master_user=master_user,
            status=EntityStatus.Pending,
            defaults=data,
        )

    def __str__(self):
        return self.number


class DebitSponsor(BaseModel):
    name = models.CharField(
        choices=DebitSponsorName.choices,
        validators=[DebitSponsorName.validator],
        max_length=32,
    )
    entity_name = models.CharField(max_length=32, help_text=_("name needed in nach xml file"))
    provider = models.IntegerField(choices=UserLoanProvider.choices, default=UserLoanProvider.IIFL)
    loan_provider = models.CharField(
        choices=LoanProvider.choices,
        validators=[LoanProvider.validator],
        max_length=24,
        null=True,
    )
    utility_code = models.CharField(max_length=128)
    sponsor_code = models.CharField(max_length=128)
    npci_utility_name = models.CharField(max_length=1024)
    creditor_name = models.CharField(max_length=1024)
    creditor_bank_name = models.CharField(max_length=128)
    creditor_account_number = models.CharField(max_length=128)
    creditor_ifsc_code = models.CharField(max_length=64)
    generate_report_type = models.CharField(
        choices=ReportType.choices, max_length=32, blank=True, null=True
    )  # this will be NULL for cases where reports aren't generated, instead a Digio integration activates
    # the physical and digital NACHes
    dispatch_report_type = models.CharField(choices=ReportType.choices, max_length=32, blank=True, null=True)

    class Meta:
        unique_together = (("name", "provider"),)

    def __str__(self):
        return "{} ({})".format(self.name, self.loan_provider)

    @classmethod
    def exclude_kiosk(cls):
        # sponsor to be excluded from kiosk queues
        return cls.objects.filter(
            name=DebitSponsorName.YesBank, loan_provider=LoanProvider.FullertonSubvention
        )


class DebitInstructionQuerySet(EntityBaseQuerySet):
    def having_valid_signed_docs(self):
        return self.filter(
            (Q(esigned_doc__isnull=False) | Q(preceding__isnull=True)), signed_doc__isnull=False
        )

    def incomplete_resetup(self):
        return (
            self.valid()
            .filter(preceding__isnull=False, esigned_doc__isnull=True)
            .exclude(nach__status=NachStatus.Cancelled)
        )

    def first_resetup(self):
        return self.incomplete_resetup().filter(status=EntityStatus.Frozen)

    def final_resetup(self):
        return self.incomplete_resetup().filter(status=EntityStatus.Rejected)

    def valid_dispatch(self):
        return self.exclude(sponsor__name__in=["bill_desk", "icici"])

    def exclude_sponsor(self):
        # excluding sponsor not to be shown in kiosk queues.
        return self.exclude(sponsor__in=DebitSponsor.exclude_kiosk())

    def exclude_formatted_kotak_mandates(self):
        # Exclude formatted NACHes registered on Kotak as sponsor
        return self.exclude(sponsor__name=DebitSponsorName.KotakMahindra)


class DebitInstructionHistory(
    metaclass=get_status_history_mcls("DebitInstructionHistory", "users.DebitInstruction")
):
    pass


@base_reversion_register()
class DebitInstruction(EntityMixin, BaseDebitInstruction):
    HISTORY_STORE = DebitInstructionHistory
    __original_signed_doc_id = None
    master_user = models.ForeignKey(MasterUser)
    sponsor = models.ForeignKey(DebitSponsor)
    bank_account = models.ForeignKey("users.BankAccount", null=True, blank=True, related_name="dis")

    generated_doc = models.OneToOneField(
        "docs.Document",
        null=True,
        editable=False,
        related_name="generated_debit_instruction",
        on_delete=models.SET_NULL,
    )
    signed_doc = models.OneToOneField(
        "docs.Document",
        null=True,
        editable=False,
        related_name="signed_debit_instruction",
        on_delete=models.SET_NULL,
    )
    formatted_doc = models.OneToOneField(
        "docs.Document",
        null=True,
        editable=False,
        related_name="formatted_debit_instruction",
        on_delete=models.SET_NULL,
    )
    esigned_doc = models.OneToOneField(
        "docs.Document",
        null=True,
        editable=False,
        related_name="esigned_debit_instruction",
        on_delete=models.SET_NULL,
    )
    courier_preferred = models.NullBooleanField(blank=True)
    source = models.ForeignKey(Source, null=True, blank=True)

    preceding = models.ForeignKey(
        "DebitInstruction",
        null=True,
        blank=True,
        help_text=_(
            "Setting this option will make this debit instruction a resetup."
            " If the corresponding nach is in non cancelled state, "
            "preceding one should never be used."
        ),
    )
    revert_nach_comments = models.TextField(blank=True, null=True)
    loan_application = models.ManyToManyField(LoanApplication, blank=True, related_name="dis")

    objects = EntityBaseManager.from_queryset(DebitInstructionQuerySet)()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_esigned_doc_id = self.esigned_doc_id

    def __str__(self):
        return "{} ({})".format(self.id, self.sponsor.name)

    def save(self, *args, **kwargs):
        if self.id is None:
            LOGGER.info(
                f"Nach for master_user_id:{self.master_user_id}, "
                "debit_instruction_id:None, nach_id:None "
                "called from debit_instruction save() during creation"
            )
        return super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        if self.nach.status in [NachStatus.InProcess, NachStatus.SentForRegistration, NachStatus.Frozen]:
            raise ValidationError(
                "Debit instruction can't be deleted if the corresponding NACH is in state InProcess ,"
                " SentForRegistration or Frozen"
            )
        return super().delete(*args, **kwargs)

    @property
    def is_resetup(self):
        """
        tells If this debit instruction is created for nach reset up purpose.
        If this function change, consider changing having_valid_signed_docs
        :return: Boolean
        """
        return self.preceding is not None

    @property
    def has_valid_signed_doc(self):
        """
        if: debit instruction is_resetup than return False if esign is not done
        else: return according to signed_doc
        :return: Boolean
        """
        if self.is_resetup and not self.esigned_doc:
            return False
        else:
            return bool(self.signed_doc)

    @property
    def has_verified_signed_doc(self):
        if self.signed_doc is not None:
            return self.signed_doc.verified_at is not None and self.signed_doc.rejected_at is None
        return False

    @property
    def is_e_nach(self):
        return self.instruction_type in [DebitInstructionType.ENACH_DIGIO, DebitInstructionType.ENACH_NPCI]

    @property
    def is_physical(self):
        return self.instruction_type == DebitInstructionType.NACH

    @property
    def is_digital(self):
        return self.instruction_type == DebitInstructionType.NACH_DIGITAL

    @property
    def pending_at_user(self):
        if not self.instruction_type:
            return True
        elif self.is_e_nach:
            return self.nach.status not in [
                NachStatus.Active,
                NachStatus.SentForRegistration,
                NachStatus.Rejected,
            ]
        elif self.is_digital:
            return not self.signed_doc or self.signed_doc.status == DocumentStatus.UserApprovalPending
        return False

    @property
    def is_nach_completed(self):
        if not self.instruction_type:
            return False
        elif self.is_e_nach:
            return self.nach.status in [NachStatus.Active] and self.nach.umrn
        elif self.is_physical and self.signed_doc:
            return True
        elif (
            self.is_digital
            and self.signed_doc
            and self.signed_doc.status != DocumentStatus.UserApprovalPending
        ):
            return True
        return False

    @cached_property
    def is_bill_desk(self):
        return self.sponsor.name == DebitSponsorName.BillDesk

    @cached_property
    def is_yes_bank(self):
        return self.sponsor.name == DebitSponsorName.YesBank

    @property
    def has_nach_changed(self):
        existing_nach = getattr(self, "nach", None)

        nach = NACH.build_from_bank_account(
            self.bank_account, self, nach_id=getattr(existing_nach, "pk", None)
        )

        di_nach_raw = existing_nach.raw if existing_nach else None
        is_same = (di_nach_raw == nach.raw or self.has_verified_signed_doc) and (
            self.instruction_type != DebitInstructionType.NACH or self.generated_doc_id
        )
        if not is_same:
            LOGGER.info(
                f"Reset debit instruction for master_user_id:{self.master_user_id}, "
                f"debit_instruction_id:{self.pk}, "
                f"is_resetup:{self.is_resetup}, "
                f"has_verified_signed_doc:{self.has_verified_signed_doc}, "
                f"instruction_type:{self.instruction_type},"
                f"generated_doc_id:{self.generated_doc_id}, "
                f"debit_instruction.nach.raw:{di_nach_raw}, "
                f"nach.raw:{nach.raw}"
            )
        return not is_same

    def generate_esigned_doc(self):
        """
        Add empty page in start of nach pdf

        Returns (ContentFile): first page is blank and 2nd one is nach

        """
        pdf = PdfFileReader(self.generated_doc.file)
        output = PdfFileWriter()
        output.addBlankPage(width=600, height=500)
        output.appendPagesFromReader(pdf)
        buffer = io.BytesIO()
        output.write(buffer)
        buffer.seek(0)
        return ContentFile(buffer.read())

    def save_esigned_doc(self, doc, doc_repo_id):
        self.esigned_doc = Document.save_nach(
            self,
            ContentFile(doc),
            signed=True,
            signature_type=SignatureType.Digital,
            doc_repo_id=doc_repo_id,
        )

    def can_complete(self):
        has_signed_agreement = self.signed_doc_id is not None
        return has_signed_agreement or self.courier_preferred is True

    def can_submit(self):
        return self.can_complete()

    def can_freeze(self):
        return bool(self.pk)

    @transition(
        "status",
        source=(EntityStatus.Approved,),
        target=EntityStatus.Frozen,
        conditions=[can_freeze],
    )
    def freeze(self):
        self.nach.freeze()
        self.nach.save()

    def can_unfreeze(self):
        return bool(self.pk)

    @transition(
        "status",
        source=(EntityStatus.Frozen,),
        target=EntityStatus.Approved,
        conditions=[can_unfreeze],
    )
    def unfreeze(self):
        # needed only for nach reset up
        self.is_valid = True
        self.nach.unfreeze()
        self.nach.save()

    def can_reject(self):
        return bool(self.pk)

    @transition(
        "status",
        source=(
            EntityStatus.PreApproved,
            EntityStatus.Submitted,
            EntityStatus.Pending,
            EntityStatus.Approved,
            EntityStatus.Frozen,
        ),
        target=EntityStatus.Rejected,
        conditions=[can_reject],
    )
    def reject(self, auth_user=None):
        pass

    class Meta:
        indexes = [
            models.Index(fields=["updated_at"]),
        ]


@base_reversion_register()
class Report(BaseModel):
    FSM_FIELDS = ("status",)

    file = models.FileField(upload_to="reports", null=True, blank=True)
    kind = models.CharField(choices=ReportType.choices, max_length=32)
    status = IntegerFSMField(
        choices=ReportStatus.choices,
        default=ReportStatus.Created,
        validators=[ReportStatus.validator],
        protected=True,
    )
    sent_at = models.DateTimeField(blank=True, null=True)

    # reference has unique index when not null
    reference = models.CharField(blank=True, null=True, max_length=32)
    parent = models.ForeignKey(
        "Report",
        blank=True,
        null=True,
        related_name="children",
        help_text=_(
            "parent report, for yes_bank_nach_input_zip, parent will beyes_bank_nach_generated type"
        ),
    )

    def __str__(self):
        return "{}:{}".format(self.id, self.file.name)

    @transaction.atomic()
    def save(self, *args, **kwargs):
        from users.tasks import create_subsequent_base

        if not self.id and self.kind == ReportType.subsequent_user_base:
            transaction.on_commit(lambda: create_subsequent_base.delay(self.id))

        super().save(*args, **kwargs)

    @cached_property
    def zip_report(self):
        zip_report_type = ReportType.get_zip_report_type(self.kind)

        if zip_report_type:
            return Report.objects.filter(kind=zip_report_type, parent=self).first()

    def to_dict(self):
        data = self.__dict__
        del data["_state"]
        for key in ["created_at", "updated_at"]:
            data[key] = str(data[key])
        data["kind"] = self.get_kind_display()
        data["file"] = self.file.url
        return data

    def can_send(self):
        allowed_report_type = ReportType.generated() + ReportType.dispatched()
        return self.status != ReportStatus.Sent and self.kind in allowed_report_type

    @transition(
        field=status,
        conditions=[can_send],
        source=[ReportStatus.Created, ReportStatus.InProcess],
        target=ReportStatus.Sent,
    )
    def send(self, reference=None):
        self.sent_at = utils.now()
        if reference:
            self.reference = reference

    @transition(field=status, source=ReportStatus.Created, target=ReportStatus.InProcess)
    def in_process(self):
        pass


class NACHStatusHistory(BaseModel):
    nach = models.ForeignKey(
        "NACH",
        related_name="transitions",
        on_delete=models.CASCADE,
    )
    status = models.PositiveSmallIntegerField(choices=NachStatus.choices)
    rejection_reason = models.ForeignKey(
        "NachRejectionReason", null=True, blank=True, on_delete=models.CASCADE
    )
    remarks = models.TextField(blank=True, null=True)
    ts = models.DateTimeField(auto_now_add=True)
    data_upload_date = models.DateField(editable=False, null=True, blank=True)
    data_approved_on = models.DateField(editable=False, null=True, blank=True)
    npci_acknowledge_date = models.DateField(editable=False, null=True, blank=True)
    npci_response_date = models.DateField(editable=False, null=True, blank=True)
    batch_id = models.CharField(max_length=128, blank=True, null=True)
    processor_unique_no = models.CharField(max_length=128, blank=True, null=True)
    status_received_from_sponsor = models.TextField(blank=True, null=True)

    @classmethod
    def store_transition(cls, nach, status, rejection_reason=None, remarks=None, history_store_params=None):
        if history_store_params:
            obj = cls.objects.create(
                nach=nach,
                status=status,
                rejection_reason=rejection_reason,
                remarks=remarks,
                data_upload_date=history_store_params["data_upload_date"],
                data_approved_on=history_store_params["data_approved_on"],
                npci_acknowledge_date=history_store_params["npci_acknowledge_date"],
                npci_response_date=history_store_params["npci_response_date"],
                batch_id=history_store_params["batch_id"],
                processor_unique_no=history_store_params["processor_unique_no"],
                status_received_from_sponsor=history_store_params["status_received_from_sponsor"],
            )
        else:
            obj = cls.objects.create(
                nach=nach, status=status, rejection_reason=rejection_reason, remarks=remarks
            )
        return obj

    def __str__(self):
        return "%s (%s)" % (self.get_status_display(), self.ts)

    def add_remarks(self, remarks):
        self.remarks = "|".join([self.remarks or "", remarks]).strip("|")


class NACHQuerySet(QuerySet):

    TIMESTAMPS = {
        "unformatted_ts": NachStatus.Unformatted,
        "new_ts": NachStatus.New,
        "inprocess_ts": NachStatus.InProcess,
        "sent_for_registration_ts": NachStatus.SentForRegistration,
        "active_ts": NachStatus.Active,
        "soft_rejected_ts": NachStatus.SoftRejected,
        "frozen_ts": NachStatus.Frozen,
        "rejected_ts": NachStatus.Rejected,
        "cancelled_ts": NachStatus.Cancelled,
        "expired_ts": NachStatus.Expired,
    }

    def with_timestamps(self, target_statuses=None):
        annotation_map = {
            timestamp_name: Max(
                Case(
                    When(
                        transitions__status=target_status,
                        then=F("transitions__ts"),
                    ),
                    default=None,
                )
            )
            for timestamp_name, target_status in self.TIMESTAMPS.items()
            if (target_statuses is None or target_status in target_statuses)
        }
        return self.annotate(**annotation_map)


@base_reversion_register()
class NACH(BaseModel):
    FSM_FIELDS = ("status",)
    HISTORY_STORE = NACHStatusHistory

    debit_instruction = models.OneToOneField(
        DebitInstruction,
        editable=False,
        on_delete=models.PROTECT,
    )
    start_date = models.DateField(null=True, blank=True)
    expiry_date = models.DateField(
        default=date(2099, 12, 31)
    )  # applicable for debit sponsors other than RBL
    ifsc = models.CharField(max_length=64)
    micr = models.CharField(max_length=64, null=True, blank=True)
    phone = PhoneNumberField()
    email_id = models.CharField(max_length=128)
    # umrn is unique when not null
    umrn = models.CharField(max_length=128, blank=True, null=True)
    holder_name = models.CharField(max_length=128)
    bank_name = models.CharField(max_length=256)
    account_number = models.CharField(max_length=256)
    account_type = models.PositiveSmallIntegerField(
        choices=BankAccountType.choices,
        validators=[BankAccountType.validator],
        default=BankAccountType.Savings,
    )
    amount = models.FloatField()
    status = IntegerFSMField(
        choices=NachStatus.choices,
        default=NachStatus.Unformatted,
        protected=True,
    )
    sub_status = IntegerFSMField(
        choices=NachSubStatus.choices,
        null=True,
        blank=True,
    )

    remarks = models.TextField(blank=True, null=True)
    active_at = models.DateTimeField(blank=True, null=True)
    soft_rejected_at = models.DateTimeField(blank=True, null=True)
    frozen_at = models.DateTimeField(blank=True, null=True)
    rejected_at = models.DateTimeField(blank=True, null=True)
    sent_for_registration_at = models.DateTimeField(blank=True, null=True)
    mandate_id = models.CharField(null=True, blank=True, unique=True, max_length=256)
    npci_ref_no = models.CharField(null=True, blank=True, unique=True, max_length=256)  # only for eNACHes
    try_times = models.IntegerField(null=True, blank=True)

    objects = NACHQuerySet.as_manager()

    class Meta:
        verbose_name_plural = _("Naches")
        permissions = (
            ("change_start_date", "Can change start date"),
            ("bulk_update_status", "Can bulk update status"),
        )
        indexes = [
            models.Index(fields=["updated_at"]),
        ]

    def __str__(self):
        return "{} (di {})".format(self.id, self.debit_instruction_id)

    def save(self, *args, **kwargs):
        if self.status == NachStatus.Unformatted and (
            self.debit_instruction.is_bill_desk
            or self.debit_instruction.instruction_type == DebitInstructionType.ENACH_DIGIO
        ):
            # todo save/copy doc
            self.move_to_new(store=False)
        return super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        """
        It is safe to delete nach when
        1. if debit_instruction type is not physical or digital
        2. if status is in (Unformatted, New, Active, Rejected, SoftRejected, Cancelled, Expired, Failed)
        and no disbursed application exist
        """
        if self.status in [NachStatus.InProcess, NachStatus.SentForRegistration, NachStatus.Frozen]:
            raise ValidationError(
                "Nach with state in InProcess , SentForRegistration , frozen can't be deleted"
            )
        return super().delete(*args, **kwargs)

    @property
    def phone_number(self):
        if self.phone.country_code:
            return "+{}-{}".format(self.phone.country_code, self.phone.national_number)
        return str(self.phone.national_number)

    @property
    def nach_id(self):
        return str(self.pk)

    @cached_property
    def times_sent(self):
        # No of times this nach has been sent to bill desk
        return (
            NachReportLog.objects.select_related("generated_report")
            .filter(nach_id=self.id, generated_report__status=ReportStatus.Sent)
            .count()
        )

    @property
    def is_valid_for_upload_response(self):
        return self.status in [NachStatus.Rejected, NachStatus.Active, NachStatus.SentForRegistration]

    @classmethod
    def times_sent_bulk(cls, nach_ids):

        nach_entities = (
            NachReportLog.objects.select_related("generated_report")
            .filter(nach_id__in=nach_ids, generated_report__status=ReportStatus.Sent)
            .values("nach_id")
            .annotate(times_sent=Count("nach_id"))
        )
        result = {nach_entity["nach_id"]: nach_entity["times_sent"] for nach_entity in nach_entities}
        for nach_id in nach_ids:
            if nach_id not in result:
                result[nach_id] = 0
        return result

    @classmethod
    def build_from_bank_account(cls, bank_account, debit_instruction, nach_id=None):
        from users.data.debit_instruction import get_nach_amount

        nach_amount = get_nach_amount(bank_account.master_user)

        account_type = bank_account.account_type
        if account_type == BankAccountType.JointSB:
            account_type = BankAccountType.Savings
        return cls(
            pk=nach_id,
            start_date=debit_instruction.created_at.date(),
            ifsc=bank_account.branch.ifsc,
            micr=bank_account.branch.micr,
            phone=bank_account.master_user.phone.phone,
            email_id=bank_account.master_user.email,
            holder_name=bank_account.holder_name,
            bank_name=bank_account.branch.bank.name,
            account_number=bank_account.number,
            account_type=account_type,
            amount=nach_amount,
        )

    def get_mandate_request_id(self, report=None):
        if report:
            report_id = report.id
        else:
            report_id = (
                NachReportLog.objects.filter(
                    nach_id=self.id,
                    generated_report__kind=self.debit_instruction.sponsor.generate_report_type,
                )
                .latest("created_at")
                .generated_report_id
            )
        return "RID{}NID{}".format(str(report_id).zfill(8), str(self.id).zfill(8))

    def to_dict(self):
        data = self.__dict__
        del data["_state"]
        for key in ["created_at", "updated_at", "expiry_date", "phone_number"]:
            data[key] = str(data[key])
        data["status"] = self.get_status_display()
        return data

    @classmethod
    def ts_report_data_bulk(cls, ids, add_sent_for_registration_time=False):
        ids = ids or []
        result = []
        count = 1
        for nach in cls.objects.filter(id__in=ids).order_by("id"):
            # {} wasn't ensuring the order, so used list of tuples
            order_dict = OrderedDict(
                [
                    ("Sr No", count),
                    ("Lot No", ""),
                    ("Debit Type", "MAXIMUM"),
                    ("Amount", nach.amount),
                    ("Frequency", "MONTHLY"),
                    ("Start Date", nach.created_at.strftime("%d%m%Y")),
                    ("End Date", nach.expiry_date.strftime("%d%m%Y")),
                    ("A/c Holder Name", nach.holder_name),
                    ("Client ID", nach.id),
                    ("A/c Type (Capital)", nach.get_account_type_display().upper()),
                    ("Bank Name", nach.bank_name),
                    ("TS Account No", "'" + nach.account_number),
                    ("IFSC", nach.ifsc),
                    ("NACH Ver. Status", ""),
                    ("NACH Remarks", ""),
                ]
            )
            if add_sent_for_registration_time:
                order_dict.update({"sent_on": nach.sent_for_registration_status_ts.strftime("%d%m%Y")})

            result.append(order_dict)
            count += 1
        return result

    def add_remarks(self, remarks):
        self.remarks = "|".join([self.remarks or "", remarks]).strip("|")

    def create_final_resetup(self, remark):
        """
        This will be the last time we resetup nach of a user.
        :return:
        """
        debit_instruction = self.debit_instruction
        self.add_remarks(remark)
        self.mark_rejected()
        debit_instruction.reject()
        debit_instruction.save()
        bank_account = debit_instruction.bank_account
        new_debit_instruction = DebitInstruction.objects.create(
            master_user=debit_instruction.master_user, preceding=debit_instruction, bank_account=bank_account
        )
        for application in debit_instruction.loan_application.all():
            new_debit_instruction.loan_application.add(application)
        LOGGER.info(
            f"Nach for master_user_id:{new_debit_instruction.master_user_id}, "
            f"debit_instruction_id:{new_debit_instruction.id}, nach_id:None "
            "called from debit_instruction create final setup"
        )
        create_and_link_nach(new_debit_instruction)
        if debit_instruction.master_user.should_send_comms:
            NachReSetupCommsScheduler(debit_instruction.master_user_id, NachResetupTypes.FINAL).schedule()

    def _total_nach_attempts_done(self):
        """
        Calculate total no of attempts done for nach sent including reprocess and resetup
        Returns:
            int
        """
        debit_instruction = self.debit_instruction
        times_sent = self.times_sent
        if debit_instruction.is_resetup:
            return times_sent + debit_instruction.preceding.nach.times_sent
        else:
            return times_sent

    def handle_rejection(self, remarks, rejection_reason=None, history_store_params=None):
        """
        It handles rejection logic of nach and corresponding di.
        Note that there should always be one valid debit
        instruction (if bank account info is present).

        Args:
            remarks:
            rejection_reason:

        Returns:

        """
        from users.utils.entity import create_and_link_nach

        debit_instruction = self.debit_instruction
        total_attempts_done = self._total_nach_attempts_done()

        self.mark_soft_rejected(rejection_reason=rejection_reason, history_store_params=history_store_params)

        if rejection_reason and rejection_reason.action == NachRejectionAction.Reject:
            self.add_remarks(NACH_REJECTED_BY_REJECTION_ACTION)
            self.mark_rejected(rejection_reason=rejection_reason, history_store_params=history_store_params)

        elif rejection_reason and rejection_reason.action == NachRejectionAction.ReportBug:
            msg = "Nach id {}, rejected by unexpected reason {}".format(self.id, rejection_reason)
            LOGGER.error(msg)
            self.mark_rejected(rejection_reason=rejection_reason, history_store_params=history_store_params)

        elif (
            not Loan.objects.filter(master_user_id=debit_instruction.master_user_id).active_loans().exists()
        ):
            self.add_remarks(NACH_NO_ACTIVE_LOAN_REMARKS)
            self.mark_rejected()

        elif total_attempts_done >= MAX_NACH_TOTAL_ATTEMPTS:
            # consumed maximum attempts
            self.add_remarks(NACH_TOTAL_MAX_ATTEMPT_REJECTION_REMARKS)
            self.mark_rejected()

        elif not debit_instruction.is_resetup and self.times_sent == 1:

            # This is for first time rejection.
            remarks = remarks.lower()
            if debit_instruction.is_bill_desk and any(
                keyword in remarks for keyword in CREATE_RESETUP_NACH_FOR_REMARK
            ):
                debit_instruction.freeze()
                # need to save as there can only be one valid DI per master_user_id
                debit_instruction.save()

                bank_account = debit_instruction.bank_account
                # creating new debit instruction for bank account
                self.add_remarks(CREATE_RESETUP_REMARKS)
                new_debit_instruction = DebitInstruction.objects.create(
                    master_user=debit_instruction.master_user,
                    preceding=debit_instruction,
                    bank_account=bank_account,
                )
                for application in debit_instruction.loan_application.all():
                    new_debit_instruction.loan_application.add(application)
                create_and_link_nach(new_debit_instruction)
                # changing bank account will create new
                # debit instruction and new nach in post save signal if not already created
                if debit_instruction.master_user.should_send_comms:
                    NachReSetupCommsScheduler(
                        debit_instruction.master_user_id, NachResetupTypes.FIRST
                    ).schedule()

        elif not debit_instruction.is_resetup and self.times_sent == MAX_NACH_REPROCESS_COUNT:
            # if rejection reason is in CREATE_RESETUP_NACH_FOR_REMARK,
            # then it already has invalid di, waited for 7 days and got cancelled.
            # Otherwise the nach has been sent to bill desk
            # MAX_NACH_REPROCESS_COUNT times. So creating new di and rejecting old one.
            self.create_final_resetup(NACH_MAX_REPROCESS_REJECTION_REMARKS)

        # send communication for first NACH rejection
        from products.choices import LoanApplicationStatus

        does_disbursed_application_exist = debit_instruction.loan_application.filter(
            status=LoanApplicationStatus.Disbursed
        ).exists()
        if does_disbursed_application_exist and not debit_instruction.is_bill_desk and self.times_sent == 1:
            from comms.tasks import send_communication
            from comms.choices import SMSTemplateName

            data = {"dashboard_link": settings.DASHBOARD_SHORT_URL}
            send_communication.delay(
                [debit_instruction.master_user_id], sms_template_name=SMSTemplateName.NachResetup, data=data
            )

    @property
    def raw(self):
        return {
            "account_number": self.account_number,
            "bank_name": self.bank_name,
            "ifsc": self.ifsc,
            "holder_name": self.holder_name,
            "account_type": self.account_type,
        }

    @transition(
        field=status,
        source=NachStatus.Unformatted,
        target=NachStatus.New,
    )
    def move_to_new(self, store=True):
        if store:
            self.HISTORY_STORE.store_transition(self, NachStatus.New)

    def can_restore_to_unformatted(self):
        debit_instruction = self.debit_instruction
        return debit_instruction.sponsor.name != DebitSponsorName.BillDesk and self.status == NachStatus.New

    @transition(
        field=status,
        source=NachStatus.New,
        target=NachStatus.Unformatted,
        conditions=[can_restore_to_unformatted],
    )
    def restore_to_unformatted(self):
        self.HISTORY_STORE.store_transition(self, NachStatus.Unformatted)

    @transition(
        field=status,
        source=[NachStatus.New, NachStatus.SoftRejected],
        target=NachStatus.InProcess,
    )
    def send_in_process(self):
        self.HISTORY_STORE.store_transition(self, NachStatus.InProcess)

    @transition(
        field=status,
        source=NachStatus.InProcess,
        target=NachStatus.New,
    )
    def restore_from_in_process(self):
        self.HISTORY_STORE.store_transition(self, NachStatus.New)

    @transition(
        field=status,
        source=NachStatus.InProcess,
        target=NachStatus.SoftRejected,
    )
    def restore_from_in_process_to_soft_rejected(self):
        # after deleting report, nach state should be put back to rejected
        # if the user was in rejected state.
        self.HISTORY_STORE.store_transition(self, NachStatus.SoftRejected)

    @transition(
        field=status,
        source=NachStatus.InProcess,
        target=NachStatus.SentForRegistration,
    )
    def send_for_registration(self):
        self.HISTORY_STORE.store_transition(self, NachStatus.SentForRegistration)
        self.sent_for_registration_at = utils.now()

    @transition(
        field=status,
        source=[NachStatus.SentForRegistration, NachStatus.InProcess],
        target=NachStatus.Active,
    )
    def mark_active(self, history_store_params=None, active_at=None):
        self.HISTORY_STORE.store_transition(
            self, NachStatus.Active, history_store_params=history_store_params
        )
        self.active_at = active_at if active_at else utils.now()

        loan_application = self.debit_instruction.loan_application.order_by("-created_at").first()
        # Incase of Repeat(same) loan for KSF and PayUFin,if nach is active and loan is disbursed too,
        # we will map new di with old application
        if (
            loan_application
            and loan_application.is_provider_payufin_or_ksf()
            and loan_application.is_subsequent_with_same_lender
        ):
            from users.utils.entity import link_previous_applications_with_new_di

            link_previous_applications_with_new_di(loan_application=loan_application, nach=self)

    @transition(
        field=status,
        source=[NachStatus.SentForRegistration, NachStatus.Active, NachStatus.Frozen],
        target=NachStatus.SoftRejected,
    )
    def mark_soft_rejected(self, rejection_reason=None, history_store_params=None):
        self.HISTORY_STORE.store_transition(
            self,
            NachStatus.SoftRejected,
            rejection_reason=rejection_reason,
            history_store_params=history_store_params,
        )
        self.soft_rejected_at = utils.now()

    @transition(
        field=status,
        source=[NachStatus.SoftRejected],
        target=NachStatus.Frozen,
    )
    def freeze(self):
        # It should be marked frozen when it is soft rejected and a
        # new nach is created and waiting for new nach to get
        # signed
        self.HISTORY_STORE.store_transition(self, NachStatus.Frozen)
        self.frozen_at = utils.now()

    @transition(
        field=status,
        source=[NachStatus.Frozen],
        target=NachStatus.SoftRejected,
    )
    def unfreeze(self):
        self.HISTORY_STORE.store_transition(self, NachStatus.SoftRejected)
        self.add_remarks("unfreeze")

    @transition(
        field=status,
        source=[NachStatus.SoftRejected, NachStatus.Frozen],
        target=NachStatus.Rejected,
    )
    def mark_rejected(self, rejection_reason=None, history_store_params=None):
        self.HISTORY_STORE.store_transition(
            self,
            NachStatus.Rejected,
            rejection_reason=rejection_reason,
            history_store_params=history_store_params,
        )
        self.rejected_at = utils.now()

    @transition(
        field=status,
        source=[NachStatus.New, NachStatus.Frozen, NachStatus.Active],
        target=NachStatus.Cancelled,
    )
    def mark_cancelled(self):
        # Once cancelled, status can't be changed
        self.HISTORY_STORE.store_transition(self, NachStatus.Cancelled)

    @transition(
        field=status,
        source=NachStatus.Rejected,
        target=NachStatus.SoftRejected,
    )
    def move_rejected_to_soft_rejected(self):
        self.HISTORY_STORE.store_transition(self, NachStatus.SoftRejected)

    @transition(
        field=status,
        source=[NachStatus.SentForRegistration, NachStatus.InProcess],
        target=NachStatus.Rejected,
    )
    def rejected(self, rejection_reason=None, remarks=None):
        self.HISTORY_STORE.store_transition(
            self, NachStatus.Rejected, rejection_reason=rejection_reason, remarks=remarks
        )
        self.rejected_at = utils.now()

    @transition(
        field=status,
        source=NachStatus.New,
        target=NachStatus.SentForRegistration,
    )
    def move_new_to_sent_for_registration(self, sent_for_registration_at=None):
        self.HISTORY_STORE.store_transition(self, NachStatus.SentForRegistration)
        self.sent_for_registration_at = sent_for_registration_at if sent_for_registration_at else utils.now()

    @transition(
        field=status,
        source=NachStatus.SentForRegistration,
        target=NachStatus.Failed,
    )
    def mark_failed(self, remarks=None):
        self.HISTORY_STORE.store_transition(self, NachStatus.Failed, remarks=remarks)


class NachRejectionReason(BaseModel):
    """
    For eNACHes, this represents the rejection reason for a failed transaction
    For physical and digital NACHes, this represents the rejection reason by the destination bank
    """

    code = models.CharField(max_length=16, unique=True, null=True, blank=True)
    name = models.CharField(max_length=1024)
    description = models.TextField(blank=True, null=True)
    action = models.IntegerField(
        choices=NachRejectionAction.choices, validators=[NachRejectionAction.validator]
    )
    sponsor_name = models.CharField(
        choices=DebitSponsorName.choices,
        validators=[DebitSponsorName.validator],
        max_length=32,
    )

    def __str__(self):
        return "{}-{} ({})".format(self.code, self.name, self.sponsor_name)


class BankAccountHistory(metaclass=get_status_history_mcls("BankAccountHistory", "users.BankAccount")):
    pass


@base_reversion_register
class NachReportLog(BaseModel):
    nach = models.ForeignKey(NACH)
    generated_report = models.ForeignKey(Report, blank=True, null=True, related_name="generated_reports")
    response_report = models.ForeignKey(Report, blank=True, null=True, related_name="response_reports")


@base_reversion_register()
class BankAccount(EntityMixin, BaseBankAccount, ExternalAdditionalDataMixin):
    HISTORY_STORE = BankAccountHistory
    master_user = models.ForeignKey(MasterUser)
    debit_instruction = models.OneToOneField(
        DebitInstruction,
        null=True,
        editable=False,
        on_delete=models.SET_NULL,
    )
    loan_application = models.ManyToManyField(LoanApplication, blank=True, related_name="bank_accounts")

    def can_complete(self):
        return self.pk is not None

    def can_submit(self):
        return self.can_complete()

    def can_reject(self):
        return bool(self.pk)

    @transition(
        "status",
        source=(
            EntityStatus.Submitted,
            EntityStatus.PreApproved,
            EntityStatus.Pending,
        ),
        target=EntityStatus.Rejected,
        conditions=[can_reject],
    )
    def reject(self, auth_user=None):
        for di in self.dis.valid():
            di.reject(auth_user=auth_user)
            di.save()

    @transaction.atomic
    def invalidate(self):
        for di in self.dis.valid():
            di.invalidate()
            di.save()
        super().invalidate()

    @property
    def is_digio_enach_supported(self):
        if self.dis.valid().filter(instruction_type=DebitInstructionType.ENACH_DIGIO).exists():
            return True
        non_digio_provider_obj = CoreGlobalConfig.objects.filter(key=NON_DIGIO_USER_LOAN_PROVIDER).first()

        if non_digio_provider_obj:
            try:
                non_digio_loan_providers = json.loads(non_digio_provider_obj.value)
                application = self.master_user.latest_application
                if application and PROVIDER_REVERSE_MAP[application.provider] in non_digio_loan_providers:
                    return False
            except JSONDecodeError:
                pass

        emandate_bank = EmandateDestinationBanks.objects.filter(bank=self.branch.bank).first()
        if (
            emandate_bank
            and emandate_bank.active
            and emandate_bank.api_mandate
            and not self.master_user.is_byju
        ):
            return True
        return False

    class Meta:
        permissions = (
            ("change_bankaccount_digital", "Can change bankaccount for digital cases"),
            ("admin_edit_bank", "Kiosk user has access to edit bank"),
        )
        indexes = [
            models.Index(fields=["updated_at"]),
        ]


class EmploymentHistory(metaclass=get_status_history_mcls("EmploymentHistory", "users.Employment")):
    pass


@base_reversion_register()
class Employment(EntityMixin, BaseModel):
    HISTORY_STORE = EmploymentHistory
    employer = models.ForeignKey("entities.Employer", blank=True, null=True)
    employer_name = models.CharField(max_length=255, blank=True, null=True)
    employment_type = models.CharField(
        choices=EmploymentType.choices,
        validators=[EmploymentType.validator],
        max_length=32,
        blank=True,
        null=True,
    )
    monthly_income = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    income_as_on_payslip = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        editable=False,
        null=True,
    )
    payment_mode = models.CharField(
        choices=PaymentMode.choices,
        validators=[PaymentMode.validator],
        max_length=24,
        blank=True,
    )
    email = models.ForeignKey(Email, null=True, blank=True)
    bank = models.ForeignKey(Bank, blank=True, null=True)
    master_user = models.ForeignKey(MasterUser)
    # For self-employed
    industry_subcategory = models.ForeignKey(
        "entities.IndustrySubcategory",
        blank=True,
        null=True,
    )
    # RBL-specific (for now)
    employer_type = models.PositiveSmallIntegerField(
        choices=SalariedEmployerType.choices,
        blank=True,
        null=True,
    )  # for salaried
    employment_category = models.PositiveSmallIntegerField(
        choices=SelfEmploymentCategory.choices,
        blank=True,
        null=True,
    )  # for self-employed
    employment_duration_in_months = models.PositiveIntegerField(
        blank=True,
        null=True,
    )  # for northern-arc
    loan_application = models.ManyToManyField(LoanApplication, blank=True, related_name="employments")
    official_email_verification_status = models.PositiveSmallIntegerField(
        choices=OfficialEmailVerificationStatus.choices, default=OfficialEmailVerificationStatus.NotInitiated
    )
    no_work_email_id = models.NullBooleanField()

    # True if
    #       1. official_email_verification_status is Valid
    #       2. Or Karza API returns non empty response with either isEmployerVerifiedViaPF or isEmployerVerifiedViaEmail as True
    # False if
    #       1. Karza API returns empty response
    #       2. Or official_email_verification_status is Invalid except when employer is verified via pf
    #       3. Or Karza API returns response with isEmployerVerifiedViaPF and isEmployerVerifiedViaEmail as None
    is_employer_verified = models.NullBooleanField(blank=True, null=True)
    employer_verification_method = models.PositiveSmallIntegerField(
        choices=EmployerVerificationMethod.choices,
        null=True,
        blank=True,
    )

    @property
    def bank_name(self):
        if self.bank_id is not None:
            return self.bank.name

    def is_self_employed(self):
        return self.employment_type == EmploymentType.SelfEmployed

    def is_salaried(self):
        return self.employment_type == EmploymentType.Salaried

    def has_rbl_details(self):
        if self.is_salaried():
            return self.employer_type is not None
        elif self.is_self_employed():
            return self.employment_category is not None

    def can_complete(self):
        has_employer = bool(self.employer_name or self.employer)
        has_bank = self.bank_id is not None
        is_self_employed = self.employment_type == EmploymentType.SelfEmployed
        latest_application = self.master_user.latest_application
        if latest_application and latest_application.provider_is_rbl() and not self.has_rbl_details():
            return False

        application = self.loan_application.order_by("-created_at").first()
        if AssessmentServiceWrapper(application).is_step_required(SkipStep.Employment):
            is_skippable = True
        else:
            is_skippable = False

        if is_skippable:
            return True
        else:
            return (
                self.employment_type is not None and (has_employer or is_self_employed or True) and has_bank
            )

    def can_submit(self):
        return self.can_complete()

    @classmethod
    def update_or_create_pending(cls, master_user, data):
        return cls.objects.update_or_create(
            master_user=master_user,
            status=EntityStatus.Pending,
            defaults=data,
        )

    @transition(
        "status",
        source=(
            EntityStatus.Pending,
            EntityStatus.Completed,
            EntityStatus.Submitted,
            EntityStatus.PreApproved,
        ),
        target=EntityStatus.Pending,
    )
    def reset_to_pending(self, auth_user=None):
        self.employer_verification_method = None
        self.is_employer_verified = None
        self.official_email_verification_status = OfficialEmailVerificationStatus.NotInitiated
        self.is_verification_skipped = False

        self.HISTORY_STORE.store_transition(self, EntityStatus.Pending, auth_user=auth_user)

    def __str__(self):
        return "{} @ {}".format(self.monthly_income, self.employer)

    class Meta:
        indexes = [
            models.Index(fields=["updated_at"]),
        ]


class AddressHistory(metaclass=get_status_history_mcls("AddressHistory", "users.Address")):
    pass


class AddressQuerySet(EntityBaseQuerySet):
    def current(self):
        return self.filter(address_type=AddressType.Current)

    def permanent(self):
        return self.filter(
            Q(address_type=AddressType.Permanent)
            | Q(address_type=AddressType.Current, aadhaar__isnull=False)
        )

    def office(self):
        return self.filter(address_type=AddressType.Office)

    def owned(self):
        return self.filter(address_type=AddressType.Owned)


@base_reversion_register()
class Address(EntityMixin, BaseAddress, ExternalAdditionalDataMixin, EntityPublishedAtBaseModel):
    HISTORY_STORE = AddressHistory
    RELATED_FIELDS = ("linked",)
    address_type = models.PositiveSmallIntegerField(
        choices=AddressType.choices,
        default=AddressType.Current,
        validators=[AddressType.validator],
    )
    master_user = models.ForeignKey(MasterUser)
    location_verified = models.NullBooleanField(blank=True)
    location = gis_models.PointField(blank=True, null=True)

    surrogate_verified = models.NullBooleanField(blank=True)
    surrogate_type = models.PositiveSmallIntegerField(
        choices=AddressSurrogateType.choices,
        validators=[AddressSurrogateType.validator],
        null=True,
        blank=True,
    )
    linked = models.ForeignKey("self", blank=True, null=True, on_delete=models.SET_NULL)
    # RBL requires unique code associated with  address proof (.e.g VoterID/Passport/DL)
    #  and the expiry date associated with the same in their CIF file
    document_unique_id = models.CharField(max_length=64, blank=True, null=True)
    document_expiry_date = models.DateField(blank=True, null=True)
    loan_application = models.ManyToManyField(LoanApplication, blank=True, related_name="addresses")

    objects = EntityBaseManager.from_queryset(AddressQuerySet)()

    @property
    def permanent_is_current(self):
        # This property only meant to be used where self.address_type=current
        if not self.is_current():
            return False

        # If user had claimed permanent=current, linked will be set
        #  (to another address entry of type=permanent) or if its an old
        #  user, the aadhaar entry will have address_id set to this instance
        return (self.linked and self.linked.is_permanent()) or self.aadhaar_set.exists()

    def unlink(self):
        # Remove FK on self to address (`linked`) or
        #  remove FK on address which has link to self
        if self.linked_id:
            self.linked_id = None
            self.save()
        elif getattr(self, "address", None) is not None:
            self.address.unlink()

    @transition(
        "status",
        source=(
            EntityStatus.Submitted,
            EntityStatus.PreApproved,
            EntityStatus.Pending,
            EntityStatus.Approved,
        ),
        target=EntityStatus.Rejected,
    )
    def reject(self, auth_user=None):
        pass

    def can_complete(self):
        return not self.has_default_line1()

    def can_submit(self):
        return self.can_complete()

    def __str__(self):
        return "{} ({})".format(self.postal_code_id, self.get_address_type_display())

    def is_permanent(self):
        return self.address_type == AddressType.Permanent

    def is_current(self):
        return self.address_type == AddressType.Current

    @classmethod
    def from_aadhaar_qr_data(cls, aadhaar_data, master_user=None):
        """
        Args:
            aadhaar_data (users.data.aadhaar.AadhaarData):
            master_user (users.models.MasterUser):
        Returns:
            Address:
        """

        postal_code = PostalCode.get_or_create(
            aadhaar_data.postal_code,
            aadhaar_data.city,
            aadhaar_data.state,
        )

        return cls(
            line1=aadhaar_data.get_address_line(),
            line2=aadhaar_data.landmark,
            postal_code=postal_code,
            city=aadhaar_data.city,
            state=aadhaar_data.state,
            master_user=master_user,
            address_type=AddressType.Permanent,
        )

    @classmethod
    def from_aadhaar_xml_data(cls, aadhaar_data, master_user):
        from products.data.loan_application import get_split_address

        postal_code = PostalCode.get_or_create(
            aadhaar_data.postal_code,
            aadhaar_data.city,
            aadhaar_data.state,
        )
        line1, line2 = get_split_address(aadhaar_data.address)
        return cls(
            line1=line1,
            line2=line2,
            postal_code=postal_code,
            city=aadhaar_data.city,
            state=aadhaar_data.state,
            master_user=master_user,
            address_type=AddressType.Permanent,
        )

    def merge(self, address, bypass=False):
        """
        Merge data from another Address object into self
        (if postal_code is same, unless bypass is True).

        Args:
            address (Address):
            bypass (boolean)
        """
        if not bypass and self.postal_code != address.postal_code:
            raise ValueError("postal_code mismatch, cannot merge!")

        self.line1 = address.line1
        self.line2 = address.line2
        self.postal_code = address.postal_code
        self.city = self.postal_code.city.name
        self.state = self.postal_code.city.state

    @classmethod
    def update_or_create_pending(cls, master_user, data, address_type):
        return cls.objects.update_or_create(
            master_user=master_user,
            status=EntityStatus.Pending,
            address_type=address_type,
            defaults=data,
        )

    @classmethod
    def is_permanent_address_linked_with_aadhaar(cls, master_user):
        application = LoanApplication.objects.for_user_v2(master_user).first()
        if application:
            permanent_address = application.addresses.valid().permanent().first()
            kyc = application.kycs.valid().filter(mode=KYCVerificationMode.EKYC_XML).first()
        else:
            permanent_address = Address.objects.for_user(master_user).permanent().first()
            kyc = KYCVerification.objects.for_user(master_user).first()
        aadhaar = kyc.aadhaar if kyc else None
        if aadhaar and permanent_address:
            return True if permanent_address and aadhaar.address == permanent_address else False
        return False

    class Meta:
        permissions = (
            ("read_locations", "Can view locations"),
            ("change_locations", "Can do edits on locations modal"),
            ("change_approve_address", "Can change approved address"),
        )
        indexes = [
            models.Index(fields=["updated_at"]),
        ]


class HouseOwnershipQueryset(QuerySet):
    def valid(self):
        return self.filter(is_valid=True)

    def for_user_id(self, master_user_id):
        return self.valid().filter(master_user_id=master_user_id)


class HouseOwnership(BaseModel):
    master_user = models.ForeignKey(MasterUser, related_name="owned_house")
    address = models.OneToOneField(Address)
    owner = models.CharField(choices=HouseOwner.choices, max_length=20)
    is_valid = models.BooleanField(default=True)
    has_no_ownership_proof = models.NullBooleanField()
    ownership_proofs = ArrayField(models.IntegerField(), null=True, blank=True, default=list())
    relationship_proofs_to_owner = ArrayField(models.IntegerField(), null=True, blank=True, default=list())

    objects = HouseOwnershipQueryset.as_manager()

    def __str__(self):
        return (
            f"{self.master_user.get_full_name()} "
            f"| {self.address.postal_code_id} "
            f"| owned by {self.get_owner_display()}"
        )

    def invalidate(self):
        self.is_valid = False

    def update_document_proof(self, document):
        if document.document_type in DocumentType.ownership_proofs():
            self.ownership_proofs.append(document.id)
        else:
            self.relationship_proofs_to_owner.append(document.id)


class ProviderUserMapping(BaseModel):
    phone = models.CharField(max_length=15, null=True, blank=True)
    provider = models.PositiveSmallIntegerField(
        choices=UserLoanProvider.choices,
    )


class UserBlacklist(BaseModel):
    master_user = models.ForeignKey(MasterUser)
    reason = models.CharField(max_length=255)
    added_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True)

    def __str__(self):
        return "%s (%s)" % (self.master_user_id, self.reason)

    @transaction.atomic
    def save(self, *args, **kwargs):
        for loan in Loan.objects.filter(
            master_user=self.master_user, provider_loan_id__isnull=True
        ).pending():
            if self.reason == "hunter rejected":
                loan.loan_application.reject(
                    comments="hunter rejected",
                    provider_rejected=True,
                    reason_ids=[RejectionReason.objects.get(text="hunter rejected").id],
                )
                loan.loan_application.save()
            else:
                loan.loan_application.reject(
                    admin_user=self.added_by,
                    comments="User blacklisted",
                    reason_ids=[RejectionReason.objects.get(text="User blacklisted").id],
                )
                loan.loan_application.save()

        return super().save(*args, **kwargs)


class MasterUserExtra(TenantAwareModel):
    master_user = models.OneToOneField(
        MasterUser,
        on_delete=models.PROTECT,
        related_name="extra",
    )
    # C-KYC related fields
    mother_name = models.CharField(max_length=128, null=True, blank=True)
    father_name = models.CharField(max_length=128, null=True, blank=True)
    marital_status = models.PositiveSmallIntegerField(
        choices=MaritalStatus.choices,
        blank=True,
        null=True,
    )
    # for RBL/banks
    educational_qualification = models.PositiveSmallIntegerField(
        choices=EducationalQualification.choices,
        blank=True,
        null=True,
    )
    category = models.PositiveSmallIntegerField(
        choices=Category.choices,
        blank=True,
        null=True,
    )
    community = models.PositiveSmallIntegerField(
        choices=Community.choices,
        blank=True,
        null=True,
    )

    # Loan related data
    apply_pf = models.NullBooleanField()
    iifl_cust_id = models.CharField(
        max_length=12,
        null=True,
        blank=True,
    )
    truecaller_name = models.CharField(max_length=128, null=True, blank=True)
    truecaller_name_match = models.NullBooleanField()
    byju_truecaller_decision = models.CharField(
        max_length=32, choices=FDStatus.choices, null=True, blank=True
    )
    agent_queue = models.ForeignKey("assistance.AgentQueue", null=True, blank=True)
    assigned_at = models.DateTimeField(null=True, blank=True)
    byju_decision = models.NullBooleanField()
    alternate_phone = models.CharField(max_length=13, null=True, blank=True)

    def has_rbl_details(self):
        return bool(
            self.educational_qualification is not None
            and self.category is not None
            and self.community is not None
        )

    def is_kyc_done(self):
        from products.services.policy_proxy import ProductPolicyProxy

        loan_application = LoanApplication.objects.first_valid_application_v2(self.master_user)
        product_policy = ProductPolicyProxy.for_application(application=loan_application)
        kyc_done = bool(
            (product_policy.skip_mother_name(self.master_user.source) or self.mother_name is not None)
            and self.father_name is not None
            and self.marital_status is not None
        )
        if not kyc_done:
            return False

        if loan_application.provider_is_rbl() and not self.has_rbl_details():
            return False

        return True

    class Meta:
        indexes = [
            models.Index(fields=["updated_at"]),
        ]


class LoginHistory(BaseModel):
    auth_user = models.ForeignKey(
        USER_MODEL, on_delete=models.CASCADE, related_name="logging_history", null=True, blank=True
    )
    is_successful = models.BooleanField(default=False)
    ip = models.GenericIPAddressField(null=True, blank=True)
    username = models.TextField(blank=True)
    action_type = models.IntegerField(choices=UserLoginAction.choices)


class FullertonExtra(BaseModel):
    master_user = models.OneToOneField(MasterUser, related_name="fullerton")
    lead_id = models.CharField(unique=True, max_length=128)
    sanction_id = models.CharField(unique=True, null=True, editable=False, max_length=128)
    lead_status = models.IntegerField(choices=FullertonLeadStatus.choices, null=True)

    def set_assign_to_pod(self):
        self.lead_status = FullertonLeadStatus.LEAD_ASSIGN_TO_POD
        self.save()

    def set_sanction_id(self, sanction_id):
        if sanction_id is None:
            raise Exception("Sanction id incorrect format.")
        self.sanction_id = sanction_id
        self.save()

    def set_lead_rejected(self):
        self.lead_status = FullertonLeadStatus.LEAD_REJECT
        self.save()

    def set_lead_status(self, status_code):
        choices = FullertonLeadStatus.choices

        for choice in choices:
            if status_code in FullertonLeadStatus.get_choice(choice[0]).fullerton_status_list:
                self.lead_status = choice[0]
        self.save()

    @DeprecationWarning
    @property
    def docs_collected(self):
        return self.lead_status == FullertonLeadStatus.LEAD_DOCUMENT_COLLECTED

    @DeprecationWarning
    def is_rejected(self):
        return self.lead_status == FullertonLeadStatus.LEAD_REJECT

    def is_lead_created(self):
        return self.lead_status == FullertonLeadStatus.LEAD_CREATE

    @DeprecationWarning
    def is_assign_to_pod(self):
        return self.lead_status == FullertonLeadStatus.LEAD_ASSIGN_TO_POD

    def __str__(self):
        return self.lead_id


class FullertonLeadCreationRequest(BaseModel):
    raw_request = models.TextField()
    raw_response = models.TextField()
    master_user = models.ForeignKey(MasterUser)
    application = models.ForeignKey(LoanApplication, null=True, blank=True)


class FullertonSubventionLeadCreationRequest(BaseModel):
    raw_request = models.TextField()
    raw_response = models.TextField(null=True, blank=True)
    application = models.ForeignKey(LoanApplication)
    hubble_id = models.CharField(max_length=50, null=True, blank=True)
    lead_id = models.CharField(max_length=50, null=True, blank=True)
    is_valid = models.BooleanField(default=True)
    hunter_match = models.NullBooleanField()
    uuid = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    status_code_key = models.IntegerField(null=True, blank=True)
    status_code_name = models.CharField(max_length=256, null=True, blank=True)
    error_message = models.TextField(null=True, blank=True)
    docs_uploaded = ArrayField(models.IntegerField(null=False), blank=True, null=True)
    embossing_done = models.NullBooleanField()

    def docs_pending_list(self, bpd_exists=None):
        docs_uploaded = self.docs_uploaded or []
        doc_ids = list(map(int, FullertonDocID))
        if bpd_exists is None:
            bpd_exists = BankPassbookData.objects.valid(self.application.master_user_id).exists()
        if not bpd_exists:
            doc_ids.remove(FullertonDocID.BANKPASSBOOK)
        misc_docs_exists = (
            self.application.docs.valid()
            .types(
                (
                    DocumentType.MarriageCertificate,
                    DocumentType.NotarizedAffidavit,
                    DocumentType.DOBDeclaration,
                    DocumentType.DualNameCertificate,
                )
            )
            .exists()
        )
        if not misc_docs_exists:
            doc_ids.remove(FullertonDocID.MISC)
        return set(doc_ids) - set(docs_uploaded)

    class Meta:
        ordering = ["-created_at"]


class SubventionLenderHistory(models.Model):
    """
    It will store lender status history
    """

    subvention_lead_request = models.ForeignKey(
        FullertonSubventionLeadCreationRequest, related_name="lender_transitions"
    )
    ts = models.DateTimeField(auto_now_add=True)
    status_code_key = models.IntegerField(null=True, blank=True)
    status_code_name = models.CharField(max_length=256, null=True, blank=True)


class IIFLExtra(BaseModel):
    FSM_FIELDS = ("doc_upload_status",)
    application = models.OneToOneField(LoanApplication, related_name="iifl")
    docs_uploaded = ArrayField(models.PositiveSmallIntegerField(), blank=True, null=True)
    doc_upload_status = IntegerFSMField(
        choices=IIFLDocUploadStatus.choices, default=IIFLDocUploadStatus.Pending, protected=True
    )

    @transition(
        doc_upload_status,
        source=[IIFLDocUploadStatus.Pending],
        target=IIFLDocUploadStatus.Uploaded,
    )
    def set_docs_uploaded(self):
        pass

    def are_docs_uploaded(self):
        return self.doc_upload_status == IIFLDocUploadStatus.Uploaded

    def get_pending_docs(self):
        docs_uploaded = self.docs_uploaded or []
        if self.application.is_subsequent():
            docs_to_upload = [IIFLDocsToUpload.LOAN_AGREEMENT.value]
        else:
            docs_to_upload = list(map(int, IIFLDocsToUpload))
            current_address_doc = get_docs_for_category(
                self.application, DocumentRequirementCategory.CurrentAddress
            )
            if not current_address_doc:
                docs_to_upload.remove(IIFLDocsToUpload.CURRENT_ADDRESS)
        return set(docs_to_upload) - set(docs_uploaded)


class Vehicle(BaseModel):
    registration_number = models.CharField(max_length=12, unique=True, blank=True, null=True)
    registration_date = models.DateField(null=True, blank=True)
    master_user = models.ForeignKey(MasterUser)
    maker_model = models.CharField(max_length=255, blank=True, null=True)
    owner_name = models.CharField(max_length=255, blank=True, null=True)
    is_valid = models.NullBooleanField(blank=True)
    vehicle_class = models.CharField(max_length=64, null=True, blank=True)
    not_available = models.NullBooleanField(blank=True)
    vehicle_value = models.BigIntegerField(blank=True, null=True)


class VehicleData(BaseModel):
    maker = models.CharField(max_length=255)
    vehicle_name = models.CharField(max_length=255)
    model_details = models.CharField(max_length=255, null=True)
    value = models.BigIntegerField()
    fallback_value = models.BigIntegerField()
    vehicle_type = models.SmallIntegerField(choices=VehicleType.choices)


def standard_decimal_number():
    return models.DecimalField(max_digits=12, decimal_places=2, blank=True, null=True)


def nullable_char_field(max_length):
    return models.CharField(max_length=max_length, null=True, blank=True)


def nullable_positive_integer():
    return models.PositiveIntegerField(null=True, blank=True)


class BankPassbookDataQueryset(QuerySet):
    def since(self, since=None):
        if since is not None:
            return self.filter(created_at__gte=since)

        return self

    def valid(self, master_user_id):
        return self.filter(master_user_id=master_user_id, is_valid=True)


class ByjuComputedLine(BaseModel):
    FSM_FIELDS = ("status",)
    status = IntegerFSMField(
        choices=BPDExternalStatus.choices,
        default=BPDExternalStatus.Pending,
    )
    approved_amount = standard_decimal_number()
    abb = standard_decimal_number()
    raw = JSONField(null=True, blank=True)
    reassessed_at = models.DateTimeField(null=True, blank=True)
    ticket_id = models.IntegerField(null=True, blank=True, help_text=_("FD Ticket ID"))

    @transition(
        status,
        source=(BPDExternalStatus.Pending, BPDExternalStatus.Approved),
        target=BPDExternalStatus.Approved,
    )
    def approve(self, amount):
        self.approved_amount = amount

    @transition(
        status,
        source=(BPDExternalStatus.Pending, BPDExternalStatus.Rejected),
        target=BPDExternalStatus.Rejected,
    )
    def reject(self):
        pass


class BankPassbookData(BaseModel):
    FSM_FIELDS = ("fraud_check_status",)
    master_user = models.ForeignKey(MasterUser)
    account_number = nullable_char_field(64)
    start_date = models.DateField(null=True, blank=True)
    end_date = models.DateField(null=True, blank=True)
    criteria_match_check = models.NullBooleanField(blank=True)
    bank_holder_name = nullable_char_field(255)
    bank_passbook = models.ForeignKey(Document)
    salary_found = models.NullBooleanField(blank=True)
    average_per_month_salary = standard_decimal_number()
    total_income = standard_decimal_number()
    total_expense = standard_decimal_number()
    employer_name = nullable_char_field(255)
    bounces = nullable_positive_integer()
    num_transactions_1_month = nullable_positive_integer()
    num_transactions_2_month = nullable_positive_integer()
    num_transactions_3_month = nullable_positive_integer()
    num_credit_txns_gt_5000_1_month = nullable_positive_integer()
    num_txns_gt_5000_3_month = nullable_positive_integer()
    amount_credited_1_month = standard_decimal_number()
    amount_credited_2_month = standard_decimal_number()
    amount_credited_3_month = standard_decimal_number()
    average_bank_balance = standard_decimal_number()

    fraud_check_status = IntegerFSMField(
        choices=BankPassbookFraudStatus.choices,
        default=BankPassbookFraudStatus.Pending,
    )

    data_entry_completed = models.BooleanField(default=False)
    is_valid = models.BooleanField(default=True)
    bcl = models.OneToOneField(ByjuComputedLine, null=True, blank=True)

    objects = BankPassbookDataQueryset.as_manager()

    @transition(
        fraud_check_status,
        source=(
            BankPassbookFraudStatus.Pending,
            BankPassbookFraudStatus.SentForVerification,
        ),
        target=BankPassbookFraudStatus.Verified,
    )
    def verify(self):
        pass

    @transition(
        fraud_check_status,
        source=(BankPassbookFraudStatus.Pending,),
        target=BankPassbookFraudStatus.SentForVerification,
    )
    def send_for_verification(self):
        pass


class CKYCQueryset(EntityBaseQuerySet):
    def unconfirmed(self):
        return self.valid().filter(is_confirmed__isnull=True)

    def confirmed(self):
        return self.valid().filter(is_confirmed=True)

    def non_rejected(self):
        return self.valid().exclude(is_confirmed=False)


class CKYC(EntityMixin, BaseModel):
    """
    master_user :           user to whom the CKYC details belong
    document_ids :          PKs in docs.models.Document which represent the documents that serve as CKYC proofs
    number :                unique 14 digit number linked with the ID proof
    base_search_type :      type of the parameter using which CKYC search was done
    base_search_value :     value of the parameter using which CKYC search was done
    details :               details fetched from the CKYC registry
    is_confirmed :          boolean to determine if the user has confirmed the address to be correct
                            remains NULL until the user has either rejected or confirmed the address
    details:                condensed KYC details of the user from the CKYC registry
    """

    master_user = models.ForeignKey(MasterUser)
    document_ids = ArrayField(models.BigIntegerField(), null=True, blank=True)
    number = models.CharField(max_length=120)
    phone = PhoneNumberField(blank=True, null=True)
    base_search_type = models.CharField(
        choices=CKYCBaseSearchType.choices, max_length=10, null=True, blank=True
    )
    base_search_value = models.CharField(max_length=50, null=True, blank=True)
    is_confirmed = models.NullBooleanField()
    details = JSONField(blank=True, null=True)

    objects = CKYCQueryset.as_manager()

    def can_submit(self):
        return self.pk is not None

    def can_complete(self):
        return False

    def confirm(self):
        self.is_confirmed = True

    def reject(self):
        self.is_confirmed = False

    @property
    def get_name(self):
        return self.details.get("search", {}).get("NAME", "") if self.details else ""

    @property
    def get_gender(self):
        return (
            self.details.get("download", {}).get("PERSONAL_DETAILS", {}).get("GENDER", "")
            if self.details
            else ""
        )

    def get_address(self, sanitize=False):
        from external.services.payufin import CKYC_CORRES_ADDRESS_DATA_POINTS

        personal_details = (
            self.details.get("download", {}).get("PERSONAL_DETAILS", {}) if self.details else {}
        )

        if not sanitize:
            addr_detail_list = []
            for addr_data_point in CKYC_CORRES_ADDRESS_DATA_POINTS:
                addr_detail_list.append(str(personal_details.get(addr_data_point, "")))
            return ", ".join(addr_detail_list)
        else:
            # sanitize it according to how external sources need it
            ret = {}
            for addr_data_point in CKYC_CORRES_ADDRESS_DATA_POINTS:
                # slice the string from index 7 to remove the "CORRES_" prefix
                ret[addr_data_point.lower()[7:]] = str(personal_details.get(addr_data_point, ""))
            return ret

    def get_doc_number(self, doc_type):
        identity_details = (
            self.details.get("download", {}).get("IDENTITY_DETAILS", {}).get("IDENTITY", [])
            if self.details
            else []
        )
        for identity in identity_details:
            if identity["IDENT_TYPE"] == doc_type:
                return identity["IDENT_NUM"]
        return ""

    def get_doc_status(self, doc_type):
        from external.services.payufin import CKYCIDStatus

        identity_details = (
            self.details.get("download", {}).get("IDENTITY_DETAILS", {}).get("IDENTITY", [])
            if self.details
            else []
        )
        for identity in identity_details:
            if identity["IDENT_TYPE"] == doc_type:
                return identity.get("IDVER_STATUS", CKYCIDStatus.UNVERIFIED)
        return ""

    def get_image_data(self, image_code):
        image_details = (
            self.details.get("download", {}).get("IMAGE_DETAILS", {}).get("IMAGE", [])
            if self.details
            else []
        )
        for image in image_details:
            if image["IMAGE_CODE"] == image_code:
                return image["IMAGE_DATA"]
        return ""

    def get_image_type(self, image_code):
        image_details = (
            self.details.get("download", {}).get("IMAGE_DETAILS", {}).get("IMAGE", [])
            if self.details
            else []
        )
        for image in image_details:
            if image["IMAGE_CODE"] == image_code:
                return image["IMAGE_TYPE"]
        return ""

    def save_perm_addr_docs(self):
        from external.services.payufin import CKYCImageType

        for image_type in CKYCImageType.perm_addr_proofs():
            image_data = self.get_image_data(image_type)
            if not image_data:
                continue
            base64_image = base64.b64decode(image_data)
            ext = self.get_image_type(image_type)
            doc_creation_kwargs = []
            for doc_type in CKYCImageType.get_choice(image_type).native_doc_types:
                name = "ckyc_" + DocumentType.get_choice(doc_type).label + "." + ext
                doc_creation_kwargs.append(
                    {
                        "master_user_id": self.master_user_id,
                        "source": DocumentSource.CKYC,
                        "file": ContentFile(base64_image, name=name),
                        "document_type": doc_type,
                    }
                )
            docs = []
            for kwargs in doc_creation_kwargs:
                docs.append(Document.objects.create(**kwargs))

            self.document_ids.extend([d.id for d in docs])
            return docs


class KYCVerificationQueryset(QuerySet):
    def for_user(self, master_user_id):
        return self.filter(mode__isnull=False, master_user_id=master_user_id, is_valid=True).order_by(
            "-created_at"
        )

    @DeprecationWarning
    def for_users(self, master_user_ids):
        return self.filter(mode__isnull=False, master_user_id__in=master_user_ids, is_valid=True).order_by(
            "-created_at"
        )

    def valid_xml(self):
        return self.filter(is_valid=True, mode=KYCVerificationMode.EKYC_XML).order_by("-created_at")

    def valid_xml_or_ckyc(self):
        return self.filter(
            is_valid=True, mode__in=(KYCVerificationMode.EKYC_XML, KYCVerificationMode.CKYC)
        ).order_by("-created_at")

    def later_xml(self):
        return self.filter(mode=None, is_valid=True, axml_proceed_option=AXMLProceedOption.Later).order_by(
            "-created_at"
        )

    def valid(self):
        return self.filter(mode__isnull=False, is_valid=True).order_by("-created_at")

    def xml_for_user(self, master_user_id):
        return self.for_user(master_user_id).filter(mode=KYCVerificationMode.EKYC_XML).first()

    def ckyc_for_user(self, master_user_id):
        return self.for_user(master_user_id).filter(mode=KYCVerificationMode.CKYC).first()


class KYCVerification(BaseModel, EntityPublishedAtBaseModel):
    master_user = models.ForeignKey(MasterUser)
    aadhaar = models.ForeignKey(Aadhaar, null=True, blank=True)
    mode = models.PositiveSmallIntegerField(
        choices=KYCVerificationMode.choices,
        default=KYCVerificationMode.Courier,
        validators=[KYCVerificationMode.validator],
        null=True,
        blank=True,
    )
    is_valid = models.BooleanField(default=True)
    kl_user_id = models.CharField(max_length=120, null=True, blank=True)
    kl_request_id = models.CharField(max_length=120, null=True, blank=True)
    remarks = models.CharField(max_length=60, null=True, blank=True)
    axml_proceed_option = models.CharField(
        AXMLProceedOption.choices,
        max_length=10,
        validators=[AXMLProceedOption.validator],
        null=True,
        blank=True,
    )
    source = models.ForeignKey(Source, null=True, blank=True)
    ckyc = models.ForeignKey(CKYC, null=True, blank=True)
    info_request = models.ForeignKey(UserInfoRequest, blank=True, null=True)
    loan_application = models.ManyToManyField(LoanApplication, blank=True, related_name="kycs")

    objects = KYCVerificationQueryset.as_manager()

    def add_remarks(self, remarks):
        if remarks:
            self.remarks = "|".join([self.remarks or "", remarks]).strip("|")

    def expire(self):
        self.add_remarks("expired")
        self.invalidate()
        self.save()

    @transaction.atomic
    def invalidate(self):
        self.is_valid = False
        self.save()

    def validate(self):
        self.is_valid = True

    class Meta:
        indexes = [
            models.Index(fields=["updated_at"]),
        ]


class UserSourceChangeLog(BaseModel):
    master_user = models.ForeignKey(MasterUser)
    source = models.ForeignKey(Source, null=True, blank=True)
    status = IntegerFSMField(choices=MasterUserStatus.choices)


class UserSourceLog(BaseModel):
    master_user = models.ForeignKey(MasterUser)
    source = models.ForeignKey(Source, null=True, blank=True)
    status = models.IntegerField(choices=MasterUserStatus.choices)


class UserPartnerChangeLog(BaseModel):
    master_user = models.ForeignKey(MasterUser)
    partner = models.ForeignKey(Partner, null=True, blank=True)
    status = models.IntegerField(choices=MasterUserStatus.choices, null=True, blank=True)


class MasterUserStatusLog(models.Model):
    master_user = models.ForeignKey("MasterUser", related_name="transitions")
    status = models.PositiveSmallIntegerField(choices=MasterUserStatus.choices)
    ts = models.DateTimeField(auto_now_add=True)


class MasterUserLenderLog(models.Model):
    master_user = models.ForeignKey("MasterUser")
    provider = models.PositiveSmallIntegerField(
        choices=UserLoanProvider.choices,
    )
    ts = models.DateTimeField(auto_now_add=True)


class NorthernArcDedupedPan(BaseModel):
    master_user = models.ForeignKey(MasterUser)
    number = models.CharField(max_length=10)


class ExternalRegistration(BaseModel):
    master_user = models.ForeignKey(MasterUser, blank=True, null=True)
    source = models.ForeignKey(Source)
    phone = models.ForeignKey(
        PhoneNumber,
        blank=True,
        null=True,
        on_delete=models.SET_NULL,
    )
    email = models.CharField(max_length=120, null=True, blank=True)
    partner_user_id = models.CharField(max_length=60, blank=True, null=True)
    lead_token = models.UUIDField(default=uuid.uuid4, blank=True, null=True, unique=True)
    details = JSONField(null=True, blank=True)

    class Meta:
        unique_together = (
            "partner_user_id",
            "source",
        )


class CallIvr(BaseModel):
    call_id = models.CharField(unique=True, null=True, blank=True, max_length=64)
    phone = models.ForeignKey(PhoneNumber)
    picked_up_at = models.DateTimeField(null=True, blank=True)
    hung_up_at = models.DateTimeField(null=True, blank=True)
    num_pulse = models.SmallIntegerField(null=True, blank=True)
    retry_count = models.SmallIntegerField(null=True, blank=True)
    rsp = JSONField(null=True, blank=True)


class BaseConsent(BaseModel):
    """
    Base Model for all different types of consent to be taken from user
    Bank Statement Consent
    CBP Consent
    CKYC Consent
    """

    master_user = models.ForeignKey(MasterUser)

    class Meta:
        abstract = True


class BankStatementConsent(BaseConsent):
    """
    Stores user consent for bank statements uploaded during income verification step

    application:    the application for which income verification was attempted
    perfios_txn:    the Perfios transaction against which the consent was stored

    """

    application = models.ForeignKey(LoanApplication)
    perfios_txn = models.OneToOneField(PerfiosTransaction)

    def __str__(self):
        return "Consent for user {0} with application {1}".format(
            self.master_user.phone, self.application.pk
        )


class ConsumerPullConsent(BaseConsent):
    """
    :param Partner partner: Partner which took the consent
    :param Source source: Source which took the consent (If external API)
    :param bureau: External party that has the credit information (Credit Bureau)
    """

    partner = models.ForeignKey("partners.Partner", null=True, blank=True)
    source = models.ForeignKey("Source", null=True, blank=True)
    bureau = models.CharField(max_length=20, null=True, blank=True)


class CKYCConsent(BaseConsent):
    """
    :param Partner partner: Partner which took the consent
    :param Source source: Source which took the consent (If external API)
    :param String type: type of consent given
    """

    partner = models.ForeignKey("partners.Partner", null=True, blank=True)
    source = models.ForeignKey("Source", null=True, blank=True)
    type = models.CharField(
        max_length=50, choices=CKYCConsentType.choices, default=CKYCConsentType.ConsentPull
    )


class PayUFinLeadRequest(BaseModel):
    raw_request = models.TextField()
    raw_response = models.TextField()
    application = models.ForeignKey(LoanApplication, null=True, blank=True, db_index=True)


class PayUFinExtra(BaseModel):
    application = models.OneToOneField(LoanApplication, related_name="payufin")
    partner_lead_id = models.CharField(null=True, blank=True, max_length=64, db_index=True)
    docs_uploaded = ArrayField(models.IntegerField(null=False), blank=True, null=True)
    status = models.CharField(choices=PayUFinLeadStatus.choices, max_length=64, null=True, blank=True)
    reason = models.CharField(null=True, blank=True, max_length=64)
    loan_booking_details = JSONField(
        blank=True, null=True
    )  # these details come from the integration partner
    booking_details_uploaded = models.NullBooleanField(null=True, blank=True)


class KSFBaseModel(BaseModel):
    master_user = models.ForeignKey("users.MasterUser", db_index=True)
    raw_request = models.TextField()
    raw_response = models.TextField()

    class Meta:
        abstract = True


class KSFDedupe(KSFBaseModel):
    """
    This model stores any and all communication related to KSF Dedupe
    """


class KSFCreditPolicy(KSFBaseModel):
    """
    This model stores any and all communication related to KSF Crdit Policy
    """


class KSFLoanApplicationRequest(BaseModel):
    application = models.ForeignKey(LoanApplication, db_index=True)
    raw_request = models.TextField()
    raw_response = models.TextField()


class IDFCMetaData(BaseModel):
    """
    Stores IDFC system meta data for the user

    master_user:        ForeignKey to the user for whom meta data is stored
    request_id:         unique ID to identify each unique application's journey (IDFC currently has a 30-char limit)
    crn_no:             Customer Resource Number, a unique identifier for a user in IDFC's system
    lan:                Loan Application Number, a unique identifier for a loan in IDFC's system
    limit:              upper limit of the loan amount that the user can apply for
    drawdown_count:     number of times drawdown for the user has been called
    status:             POSIDEX status of the user from IDFC stating whether they've been approved or rejected
    rejection_reason:   reason, if any, why the user was rejected by IDFC
    is_valid:           determines if the meta data is valid. This is marked False during credit expiry
                        so that further calls to IDFC do not use the existing meta data but generate new meta data
    docs_uploaded:      array of documents that have been uploaded
    kyc_info:           condensed list of key-value pairs

    """

    master_user = models.ForeignKey(MasterUser, on_delete=models.CASCADE)
    request_id = models.CharField(max_length=50, unique=True)
    crn_no = models.BigIntegerField(blank=True, null=True)
    lan = models.BigIntegerField(unique=True, blank=True, null=True)
    limit = models.IntegerField(blank=True, null=True)
    drawdown_count = models.IntegerField(default=0)
    status = models.CharField(max_length=20, blank=True, null=True)
    rejection_reason = models.CharField(max_length=100, blank=True, null=True)
    is_valid = models.BooleanField(default=True)
    docs_uploaded = ArrayField(models.CharField(max_length=20), null=True, blank=True)
    kyc_info = JSONField(blank=True, null=True)

    def __str__(self):
        return "({3}) user ({0}) with request id {1} has status as {2}".format(
            self.master_user, self.request_id, self.status, "valid" if self.is_valid else "invalid"
        )

    @staticmethod
    def record_kyc(id_of_row, kyc_detail):
        with discrete_connection_handler() as cursor:
            cursor.execute(
                """update users_idfcmetadata
                            set updated_at = now(), kyc_info = %s
                            where id = %s;""",
                (json.dumps(kyc_detail), id_of_row),
            )

    @staticmethod
    def record_posidex_response(id_of_row, lan, crn_no, status, rejection_reason=None):
        with discrete_connection_handler() as cursor:
            cursor.execute(
                """update users_idfcmetadata
                            set lan = %s, crn_no = %s, status = %s, updated_at = now()
                            where id = %s;""",
                (lan, crn_no, status, id_of_row),
            )
            if rejection_reason:
                cursor.execute(
                    """update users_idfcmetadata
                                set rejection_reason = %s, updated_at = now()
                                where id = %s;""",
                    (rejection_reason, id_of_row),
                )

    @staticmethod
    def record_limit(id_of_row, limit):
        with discrete_connection_handler() as cursor:
            cursor.execute(
                """update users_idfcmetadata
                            set "limit" = %s, updated_at = now()
                            where id = %s;""",
                (str(int(limit)), id_of_row),
            )

    @staticmethod
    def increment_drawdown(id_of_row):
        with discrete_connection_handler() as cursor:
            cursor.execute(
                """update users_idfcmetadata
                            set drawdown_count = drawdown_count + 1, updated_at = now()
                            where id = %s""",
                (id_of_row,),
            )

    @staticmethod
    def record_drawdown_rejection(id_of_row, rejection_reason):
        with discrete_connection_handler() as cursor:
            cursor.execute(
                """update users_idfcmetadata
                            set rejection_reason = %s, status = 'REJECT', updated_at = now()
                            where id = %s;""",
                (rejection_reason, id_of_row),
            )

    @classmethod
    def invalidate(cls, master_user_id):
        cls.objects.filter(master_user_id=master_user_id).update(is_valid=False)

    def get_full_ckyc_address(self):
        return ", ".join(
            [
                str(self.kyc_info.get("CKYCCorAdd1", "")),
                str(self.kyc_info.get("CKYCCorAdd2", "")),
                str(self.kyc_info.get("CKYCCorAdd3", "")),
                str(self.kyc_info.get("CKYCCorAddCity", "")),
                str(self.kyc_info.get("CKYCCorAddDistrict", "")),
                str(self.kyc_info.get("CKYCCorAddState", "")),
                str(self.kyc_info.get("CKYCCorAddCountry", "")),
                str(self.kyc_info.get("CKYCCorAddPin", "")),
            ]
        )


class NachResponse(BaseModel):
    url = models.CharField(max_length=255)
    origin = models.PositiveSmallIntegerField(
        choices=NachOrigin.choices,
        validators=[NachOrigin.validator],
        help_text=_("source (who hit the request)"),
    )
    nach = models.ForeignKey(NACH)
    status = models.TextField(blank=True, null=True)
    raw_request = models.TextField(blank=True, null=True)
    raw_response = models.TextField(blank=True, null=True)
    remarks = models.TextField(blank=True, null=True)

    class Meta:
        abstract = True


class EnachResponse(NachResponse):
    """
    This model stores any and all communication related to eNACH with a third party client (such as Digio or NPCI)
    """


class SignedNachResponse(NachResponse):
    """
    This model stores any and all communication related to physical/digital NACHes with a third party client
    (currently just Digio)
    """


class NameMismatchDeclarationQueryset(QuerySet):
    def for_user(self, master_user_id):
        return self.filter(master_user_id=master_user_id, status=NameMismatchDeclarationStatus.Active)


class NameMismatchDeclaration(BaseModel):
    """
    This model stores the data for mismatch name declaration of the users where user has different name in PAN
    and other documents.
    mismatch_name: The name of the user in the document which is different from the PAN Name
    document_type: Type of the document like Passport, Driving License the value is picked from
    NameMismatchDeclarationDocument choice
    """

    mismatch_name = models.CharField(max_length=64)
    master_user = models.ForeignKey(MasterUser)
    document_type = models.SmallIntegerField(
        choices=NameMismatchDeclarationDocument.choices,
    )
    document_number = models.TextField()
    status = models.SmallIntegerField(
        choices=NameMismatchDeclarationStatus.choices,
        default=NameMismatchDeclarationStatus.Active,
    )

    objects = NameMismatchDeclarationQueryset.as_manager()


class ExternalAdditionalData(models.Model):
    verified_at = models.DateTimeField(null=True, blank=True)
    verification_method = models.PositiveSmallIntegerField(
        choices=VerificationType.choices, null=True, blank=True
    )
    data_source = models.PositiveSmallIntegerField(choices=DataSource.choices, null=True, blank=True)
    meta_data = JSONField(
        blank=True,
        null=True,
    )
    source = models.ForeignKey(Source)
    master_user = models.ForeignKey(MasterUser)
    reference_table = models.PositiveSmallIntegerField(
        choices=AdditionalDataReferenceTable.choices, null=True, blank=True
    )
    reference_table_id = models.BigIntegerField(null=True, blank=True)


class RefinanceForeclosureDetailsQueryset(QuerySet):
    def for_user(self, master_user):
        return self.filter(application__master_user=master_user, is_valid=True).order_by(
            "-application_id", "-created_at"
        )

    def for_application_id(self, application_id):
        return (
            self.valid()
            .filter(application_id=application_id, tradeline_id__isnull=False)
            .order_by("-created_at")
        )

    def valid(self):
        return self.filter(is_valid=True)


class RefinanceHistory(
    metaclass=get_status_history_mcls("RefinanceHistory", "users.RefinanceForeclosureDetails")
):
    pass


@base_reversion_register()
class RefinanceForeclosureDetails(EntityMixin, BaseModel):
    """
    This table includes the tradelines of the user's application along with the corresponding
    foreclosure data.
    """

    HISTORY_STORE = RefinanceHistory
    foreclosure_amount = models.FloatField(null=True, blank=True)
    lender_name = models.CharField(max_length=255, null=True, blank=True)
    bank_account_number = models.CharField(max_length=255, null=True, blank=True)
    bank_ifsc = models.CharField(max_length=255, null=True, blank=True)
    bank_name = models.CharField(max_length=255, null=True, blank=True)
    beneficiary_name = models.CharField(max_length=255, null=True, blank=True)
    payment_mode = models.PositiveSmallIntegerField(
        choices=BtTopUpPaymentOption.choices,
        default=BtTopUpPaymentOption.Cheque,
        validators=[BtTopUpPaymentOption.validator],
    )
    auth_user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True)
    application = models.ForeignKey(
        LoanApplication, on_delete=models.CASCADE, related_name="refi_foreclosures"
    )
    is_verified = models.NullBooleanField(null=True, default=None)
    tradeline_id = models.CharField(max_length=255, null=True)
    max_credit_line = models.DecimalField(decimal_places=4, max_digits=12, null=True)
    bt_loan_type = models.CharField(max_length=255, null=True)
    current_balance = models.DecimalField(decimal_places=4, max_digits=12, null=True)
    max_tenure = models.IntegerField(null=True)
    actual_loan_amount = models.DecimalField(decimal_places=4, max_digits=12, null=True)
    loan_opened_date = models.DateField(
        null=True,
        blank=True,
    )
    docs = models.ManyToManyField(Document, blank=True, related_name="refi_foreclosure_detail")

    objects = RefinanceForeclosureDetailsQueryset.as_manager()

    def invalidate(self):
        self.is_valid = False

    def can_complete(self):
        return self.foreclosure_amount and self.foreclosure_amount > 0

    def can_submit(self):
        return self.can_complete()

    # class Meta:
    #     unique_together = (('tradeline_id', 'application'),)


class AuthUserExtra(BaseModel):
    auth_user = models.OneToOneField(
        USER_MODEL,
        on_delete=models.CASCADE,
    )
    phone = models.OneToOneField(PhoneNumber, blank=True, null=True, on_delete=models.CASCADE)
    profile_blocked = models.BooleanField(default=False)


class AuthUserBlockingHistory(BaseModel):
    auth_user = models.ForeignKey(
        USER_MODEL,
        on_delete=models.CASCADE,
        related_name="blocking_history",
    )
    status = models.PositiveSmallIntegerField(
        choices=AuthUserBlockingStatus.choices,
        default=AuthUserBlockingStatus.UnBlocked,
    )
    unblocked_by = models.ForeignKey(
        USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    remarks = models.TextField(null=True, blank=True)


class WhiteListedUsersQuerySet(QuerySet):
    def valid(self):
        current_date = datetime.datetime.today()
        return self.filter(valid_from__lte=current_date, valid_to__gte=current_date).order_by("-created_at")


class WhiteListedUsers(BaseModel):
    phone_number = PhoneNumberField()
    interest_grid = models.DecimalField(blank=True, decimal_places=4, max_digits=8, null=True)
    processing_fees = models.DecimalField(blank=True, decimal_places=5, max_digits=12, null=True)
    valid_from = models.DateField()
    valid_to = models.DateField()

    objects = WhiteListedUsersQuerySet.as_manager()


class AmlApproveRejectReason(BaseModel):
    action_type = models.SmallIntegerField(
        choices=AMLActionType.choices,
        default=AMLActionType.Reject,
    )
    text = models.TextField(blank=True, null=True)


class AMLResponseData(BaseModel):
    application = models.OneToOneField(LoanApplication, primary_key=True)
    decision = models.NullBooleanField()
    reason = models.ForeignKey("users.AmlApproveRejectReason", blank=True, null=True)
    comment = models.TextField(blank=True, null=True)
    document = models.ForeignKey("docs.Document", blank=True, null=True)
    master_user = models.ForeignKey("users.MasterUser")
