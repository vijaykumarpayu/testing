import logging

from django.contrib.auth import get_user_model
from django.contrib.auth.models import User
from django.db import transaction
from django.db.models import Q

from base.utils.api import APIError
from external.services.byju import Byju
from lms.choices import LoanProvider
from partners.choices import PartnerType, PartnerRoleType
from partners.constants import BYJU_V2_FLOW_CODE
from partners.models import Partner
from partners.utils.lender_products import get_product_by_lender
from users.constants import BYJU_VENDOR_CODE
from users.constants import EXTERNAL_SOURCE_KEY
from vendors.models import Vendor
from .models import OTP, MasterUser, Source, AuthUserExtra

USER_MODEL = get_user_model()
LOGGER = logging.getLogger(__name__)


class OTPAuth:
    @transaction.atomic
    def authenticate(self, request, phone=None, code=None, **kwargs):
        qs = (
            OTP.objects.valid()
            .authentication()
            .filter(
                phone=phone,
                code=code,
            )
            .select_related("phone__masteruser__auth_user")
        )
        otp = qs.first()

        if not otp:  # invalid OTP
            return

        # when PhoneNumber is associated with MasterUser
        try:
            if otp and otp.phone.masteruser and otp.phone.masteruser.auth_user:
                otp.phone.verify()
                return otp.phone.masteruser.auth_user
        except (MasterUser.DoesNotExist, USER_MODEL.DoesNotExist) as e:
            LOGGER.warning(e)

    def get_user(self, user_id):
        try:
            masteruser = MasterUser.objects.select_related("auth_user").get(auth_user_id=user_id)
            return masteruser.auth_user
        except MasterUser.DoesNotExist:
            return


class OTPAuth2:
    """
    Backend authentication for OTP login of AuthUserExtra Users(partners and agents)
    Takes (phone no. and otp ) to authenticate
    """

    @transaction.atomic
    def authenticate(self, request, phone=None, code=None, **kwargs):
        qs = (
            OTP.objects.valid()
            .authentication()
            .filter(
                phone=phone,
                code=code,
            )
            .select_related("phone__authuserextra__auth_user")
        )
        otp = qs.first()

        if not otp:  # invalid OTP
            return

        # when PhoneNumber is associated with AuthUserExtra
        try:
            if otp and otp.phone.authuserextra and otp.phone.authuserextra.auth_user:
                otp.phone.verify()
                return otp.phone.authuserextra.auth_user
        except (AuthUserExtra.DoesNotExist, USER_MODEL.DoesNotExist) as e:
            LOGGER.warning(e)

    def get_user(self, user_id):
        try:
            auth_user_extra = AuthUserExtra.objects.select_related("auth_user").get(auth_user_id=user_id)
            return auth_user_extra.auth_user
        except AuthUserExtra.DoesNotExist:
            return


class ByjuAuth:
    def create_byju_user(
        self, username, byju_info, email, role=PartnerRoleType.SalesPerson, lender=None, product=None
    ):
        vendor = Vendor.objects.get(code=BYJU_VENDOR_CODE)
        user = User.objects.create(
            username=username,
            is_active=True,
            is_staff=True,
            is_superuser=False,
        )

        return Partner.objects.create(
            auth_user=user,
            name=byju_info["name"],
            partner_type=PartnerType.PointOfSale,
            vendor=vendor,
            email=email,
            role=role,
            lender=lender,
            product=product,
        )

    @transaction.atomic
    def authenticate(self, request, email=None, lender=None, **kwargs):
        try:
            byju_info = Byju.get_user_details(email)
        except APIError as e:
            if getattr(e.exc, "response", None) is None:
                return None

            status_code = e.exc.response.status_code
            if status_code == 400:
                User.objects.filter(username=email).update(is_active=False)

            return None

        if not byju_info:
            return None

        try:
            user = User.objects.get(username=email, partner__vendor__code=BYJU_VENDOR_CODE)
            if byju_info.get("status") != "Active":
                if user.is_active:
                    user.is_active = False
                    user.save()
                return None
            partner = user.partner
            if not user.is_active:
                user.is_active = True
                user.save()
            if lender is not None:
                product = get_product_by_lender(lender)
                if lender == BYJU_V2_FLOW_CODE:
                    lender = LoanProvider.FullertonSubvention

                partner.lender = lender
                partner.product = product
                partner.save()
        except User.DoesNotExist:
            if byju_info.get("role") == "sales_agent":
                role = PartnerRoleType.SalesPerson
            elif byju_info.get("role") == "pic":
                role = PartnerRoleType.SalesManager
            else:
                raise Exception("Invalid role")

            product = get_product_by_lender(lender)
            if lender == BYJU_V2_FLOW_CODE:
                lender = LoanProvider.FullertonSubvention

            partner = self.create_byju_user(
                email, byju_info, email, role=role, lender=lender, product=product
            )
            user = partner.auth_user
            if byju_info.get("role") == "sales_agent":
                pics = byju_info["pic_emails"]
                for pic in pics:
                    try:
                        pic_user = User.objects.get(username=pic, partner__vendor__code=BYJU_VENDOR_CODE)
                        asm = pic_user.partner
                    except User.DoesNotExist:
                        pic_info = Byju.get_user_details(pic)
                        asm = self.create_byju_user(pic, pic_info, pic, role=PartnerRoleType.SalesManager)
                    partner.pic.add(asm)
        if request:
            # request is None when login or authenticate function is called explicitly e.g. test cases.
            # set session expiry to 1 day. Byju agents have to re login atleast once a day.
            request.session.set_expiry(24 * 60 * 60)  # 1 day
        return user

    def get_user(self, user_id):
        try:
            partner_user = Partner.objects.select_related("auth_user").get(auth_user_id=user_id)
            return partner_user.auth_user
        except Partner.DoesNotExist:
            return


class ExternalAuth:
    """
    This backend should ideally only be used for users coming from /external/register-v2 API
    """

    @transaction.atomic()
    def authenticate(self, request, phone=None, source=None, **kwargs):
        """
         To authenticate registration requests coming from External Sources with skip_otp enabled.
         On successful authentication, OTP will be skipped for such MasterUser-s and
         session-token will be sent to the external partner.
        Args:
            request:
            phone: phone_id of MasterUser
            source: external partner source
            **kwargs:
         Returns:
        """

        # source would only we present in case of ExternalPartnerRegistration
        if source is None:
            return

        secret = request.META.get(EXTERNAL_SOURCE_KEY)

        if (
            Source.objects.external()
            .filter(platform_source=True, secret=secret)
            .filter(Q(platform_source=True) | Q(optimized_platform=True))
            .exists()
        ):
            pass
        elif not Source.objects.external().filter(skip_otp=True, secret=secret).exists():
            return

        try:
            master_user = MasterUser.objects.get(phone_id=phone)
            return master_user.auth_user
        except MasterUser.DoesNotExist:
            return

    def get_user(self, user_id):
        try:
            masteruser = MasterUser.objects.select_related("auth_user").get(auth_user_id=user_id)
            return masteruser.auth_user
        except MasterUser.DoesNotExist:
            return
