import csv
import io
from datetime import timedelta

from django.conf.urls import url
from django.contrib import admin, messages
from django.core.files.base import ContentFile
from django.db import transaction
from django.db.models import Count, When, Case, Sum, IntegerField
from django import forms
from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect
from fsm_admin.mixins import FSMTransitionMixin
from rest_framework import generics

from base.admin import ReadonlyAdmin
from core.restframework import UserHasPermission
from core.utils import now
from products.admin import BulkUploadHunterStatus, CsvImportForm, HunterBulkUploadError
from users.choices import MasterUserStatus, NachStatus, ReportType
from users.utils.debit_instruction.base import Base as Di_Base
from users.utils.exceptions import NACHStatusBulkUpdateError
from .models import (
    MasterUser,
    PhoneNumber,
    Email,
    Source,
    Address,
    UserBlacklist,
    DebitSponsor,
    SourceRollout,
    NachRejectionReason,
    NACH,
    Report,
    RefinanceForeclosureDetails,
    AuthUserBlockingHistory,
    AuthUserExtra,
)


class SourceRolloutAdmin(admin.ModelAdmin):
    list_display = (
        "_source",
        "employment",
        "iifl_rollout",
        "fullerton_rollout",
        "rbl_rollout",
        "northern_arc_rollout",
        "payufin_rollout",
        "ksf_rollout",
        "is_external",
    )

    @staticmethod
    def _source(obj):
        if obj.source is not None:
            return obj.source
        elif obj.source is None and obj.is_external is True:
            return "Source Fallback"
        else:
            return "Paysense"

    def has_delete_permission(self, request, obj=None):
        return False

    def get_actions(self, request):
        actions = super().get_actions(request)
        if "delete_selected" in actions:
            del actions["delete_selected"]
        return actions


class SourceAdmin(admin.ModelAdmin):
    def get_queryset(self, request):
        return (
            super()
            .get_queryset(request)
            .annotate(
                c=Count("masteruser"),
            )
        )

    def count(self, obj):
        return obj.c

    count.admin_order_field = "c"

    readonly_fields = ("secret",)
    list_display = (
        "key",
        "count",
        "skip_otp",
        "tenant",
        "landing_page_source",
    )
    list_filter = ("skip_otp",)


class AuthUserExtraAdmin(admin.ModelAdmin):
    list_display = (
        "auth_user",
        "profile_blocked",
    )
    list_editable = ("profile_blocked",)
    search_fields = (
        "=auth_user__username",
        "=auth_user__email",
        "=phone__phone",
    )
    raw_id_fields = ("auth_user", "phone")
    fieldsets = (
        (
            None,
            {
                "fields": (
                    ("phone",),
                    ("auth_user",),
                    ("profile_blocked",),
                )
            },
        ),
    )

    def get_actions(self, request):
        actions = super().get_actions(request)
        if "delete_selected" in actions:
            del actions["delete_selected"]
        return actions

    def has_delete_permission(self, request, obj=None):
        return False

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        if "profile_blocked" in form.changed_data and obj.profile_blocked is False:
            AuthUserBlockingHistory.objects.create(
                auth_user=obj.auth_user,
                unblocked_by=request.user,
            )


class MasterUserAdmin(ReadonlyAdmin, admin.ModelAdmin):
    list_filter = ("status", "created_at", "source")
    list_display = (
        "id",
        "phone_id",
        "email_id",
        "first_name",
        "status",
        "customer_id",
    )
    search_fields = (
        "=customer_id",
        "=phone__phone",
        "=email__email",
        "first_name",
        "last_name",
    )
    list_per_page = 50
    raw_id_fields = ("phone",)

    def get_readonly_fields(self, request, obj=None):
        readonly = super().get_readonly_fields(request, obj)
        # allow superuser to change phone on masteruser
        if request.user.is_superuser:
            readonly.remove("phone")
        return readonly


class PhoneNumberAdmin(ReadonlyAdmin, admin.ModelAdmin):
    list_filter = ("otp_verified_at", "created_at", "otp_skipped")
    list_display = ("phone", "otp_verified_at", "is_dnd", "otp_skipped")
    list_per_page = 50
    search_fields = ("phone",)

    def has_add_permission(self, request):
        return request.user.is_superuser

    def get_readonly_fields(self, request, obj=None):
        if not obj:
            return []
        readonly = super().get_readonly_fields(request, obj=obj)
        if request.user.is_superuser:
            readonly.remove("whatsapp_enabled")
        return readonly


class EmailAdmin(ReadonlyAdmin, admin.ModelAdmin):
    list_filter = ("verified_at", "created_at")
    list_display = ("email", "verified_at", "promotional_subscribed")
    search_fields = ("email",)
    list_per_page = 50


class EntityAdminMixin(FSMTransitionMixin):
    fsm_field = ("status",)
    search_fields = (
        "=master_user__id",
        "=master_user__loanapplication__id",
    )
    show_full_result_count = False
    list_per_page = 50

    def has_module_permission(self, request):
        return request.user.is_superuser


class AddressAdmin(EntityAdminMixin, ReadonlyAdmin, admin.ModelAdmin):
    @staticmethod
    def short_line1(obj):
        if obj:
            return "%s..." % ((obj.line1 or "")[:50],)

    list_filter = (
        "status",
        "address_type",
    )
    list_display = (
        "master_user_id",
        "postal_code_id",
        "short_line1",
        "city",
        "state",
    )
    readonly_fields = ("short_line1",)


class UserBlacklistAdmin(admin.ModelAdmin):
    list_display = ("master_user_id", "created_at", "added_by", "reason")
    search_fields = ("=master_user__id",)
    list_per_page = 30
    raw_id_fields = ("master_user",)
    readonly_fields = ("added_by",)
    ordering = ["-created_at"]

    def save_model(self, request, obj, form, change):
        obj.added_by = request.user
        return super().save_model(request, obj, form, change)


class DebitSponsorAdmin(admin.ModelAdmin):
    pass


class NACHAdmin(admin.ModelAdmin):
    change_list_template = "nach/nach_changelist.html"

    list_display = ("id", "start_date", "holder_name", "status")
    search_fields = ("id",)
    list_per_page = 30
    ordering = ["-id"]
    actions = None

    def get_readonly_fields(self, request, obj=None):
        if not obj:
            return []

        read_only_fields = []
        for field in self.opts.local_fields:
            if field.name == "start_date" and obj.status in [
                NachStatus.Unformatted,
                NachStatus.New,
                NachStatus.SoftRejected,
            ]:
                continue
            read_only_fields.append(field.name)
        return read_only_fields

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    def has_change_permission(self, request, obj=None):
        if request.user.has_perm("users.change_start_date") or request.user.is_superuser:
            return True
        else:
            return False

    def get_urls(self):
        urls = super().get_urls()
        my_urls = [
            url(r"retry-nach-upload/", NACHRetryUpload.as_view(), name="bulk_upload_nach_status"),
        ]
        return my_urls + urls


class NachRejectionReasonAdmin(admin.ModelAdmin):
    list_display = ("id", "code", "name", "get_action_display")
    search_fields = ("=id", "code")
    readonly_fields = ("id",)
    ordering = ["-id"]
    actions = None


class NACHRetryUpload(generics.ListCreateAPIView):
    permission_classes = (UserHasPermission.for_(("users.bulk_update_status",)),)

    def get(self, request, *args, **kwargs):
        form = CsvImportForm()
        payload = {"form": form}
        return render(request, "nach/bulk_upload_nach_status.html", payload)

    @transaction.atomic
    def post(self, request, *args, **kwargs):
        try:
            file = request.FILES["csv_file"]
            buffer = io.StringIO(file.read().decode())
            reader = csv.DictReader(buffer)
            headers = reader.fieldnames
            if "nach_id" not in headers:
                messages.error(request, "nach_id not present in headers")
                raise NACHStatusBulkUpdateError
            nach_ids = [row["nach_id"] for row in reader]
            if NACH.objects.filter(id__in=nach_ids).exclude(status=NachStatus.Rejected).exists():
                messages.error(request, "Some naches are not in rejected state")
                raise NACHStatusBulkUpdateError
            report = Report(kind=ReportType.nach_status_bulk_update)
            report.file.save(file.name, ContentFile(buffer.getvalue().encode()))
            for nach in NACH.objects.filter(id__in=nach_ids):
                nach.move_rejected_to_soft_rejected()
                nach.add_remarks("manually reverted to soft_rejected")
                nach.save()
            Di_Base.create_log(nach_ids, report.id)
            messages.add_message(
                request, messages.SUCCESS, f"{len(nach_ids)} naches have been reverted to soft rejected"
            )
            return redirect("..")
        except NACHStatusBulkUpdateError:
            return HttpResponseRedirect(self.request.path_info)


class RefinanceForeclosureDetailsAdmin(admin.ModelAdmin):
    list_filter = ("is_verified",)
    list_display = ("id", "application_id", "is_verified")
    list_per_page = 50
    raw_id_fields = ("application",)
    search_fields = ("id", "=application_id")


class ReportForm(forms.ModelForm):
    def clean(self):
        kind = self.cleaned_data.get("kind")
        if kind != ReportType.subsequent_user_base:
            raise forms.ValidationError("report should be of type fullerton_base_for_subsequent")

        last_report = (
            Report.objects.filter(kind=ReportType.subsequent_user_base).order_by("-created_at").first()
        )
        if last_report and last_report.created_at > now() - timedelta(days=5):
            raise forms.ValidationError("Last report was created within 15 days, can't create new report")

        return self.cleaned_data


class ReportAdmin(admin.ModelAdmin):
    list_display = ("id", "kind", "file", "status", "created_at", "updated_at")
    list_per_page = 30
    ordering = ["-id"]
    exclude = ("sent_at", "reference", "parent")
    readonly_fields = ("status",)
    actions = None
    form = ReportForm

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.filter(kind=ReportType.subsequent_user_base)

    def has_delete_permission(self, request, obj=None):
        return False


admin.site.register(MasterUser, MasterUserAdmin)
admin.site.register(AuthUserExtra, AuthUserExtraAdmin)
admin.site.register(PhoneNumber, PhoneNumberAdmin)
admin.site.register(Email, EmailAdmin)
admin.site.register(Source, SourceAdmin)
admin.site.register(SourceRollout, SourceRolloutAdmin)
admin.site.register(Address, AddressAdmin)
admin.site.register(UserBlacklist, UserBlacklistAdmin)
admin.site.register(DebitSponsor, DebitSponsorAdmin)
admin.site.register(NACH, NACHAdmin)
admin.site.register(NachRejectionReason, NachRejectionReasonAdmin)
admin.site.register(RefinanceForeclosureDetails, RefinanceForeclosureDetailsAdmin)
admin.site.register(Report, ReportAdmin)
