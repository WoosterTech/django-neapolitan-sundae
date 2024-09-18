# mypy: disable-error-code="import-untyped"
import importlib
import logging
import warnings
from enum import Enum
from gettext import gettext
from typing import TYPE_CHECKING, Any, override

from crispy_forms.layout import Submit
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.core.paginator import InvalidPage, Paginator
from django.db import models
from django.db.models.fields.related import ManyToManyField
from django.forms import Widget, modelform_factory
from django.forms import models as model_forms
from django.http import Http404, HttpResponseRedirect
from django.shortcuts import get_object_or_404, redirect
from django.template.response import TemplateResponse
from django.urls import NoReverseMatch, path, reverse, reverse_lazy
from django.utils.decorators import classonlymethod
from django.utils.functional import classproperty
from django.utils.translation import gettext as _
from django.views.generic import View
from django_filters.filterset import filterset_factory
from django_rubble.utils.enums import Icon, LibraryIcon
from django_rubble.utils.model_helpers import (
    get_model_fields,
    get_model_name,
    get_model_verbose_name_plural,
)
from django_rubble.widgets import DetailWidget
from django_tables2.tables import Table, table_factory
from pydantic import BaseModel

from neapolitan_sundae.helpers.forms import CrispyModelForm

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


# A CRUDView is a view that can perform all the CRUD operations on a model. The
# `role` attribute determines which operations are available for a given
# as_view() call.
class Role(Enum):
    LIST = "list"
    DETAIL = "detail"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"

    def handlers(self):
        match self:
            case Role.LIST:
                return {"get": "list"}
            case Role.DETAIL:
                return {"get": "detail"}
            case Role.CREATE:
                return {
                    "get": "show_form",
                    "post": "process_form",
                }
            case Role.UPDATE:
                return {
                    "get": "show_form",
                    "post": "process_form",
                }
            case Role.DELETE:
                return {
                    "get": "confirm_delete",
                    "post": "process_deletion",
                }

    def extra_initkwargs(self):
        # Provide template_name_suffix, "_list", "_detail", "_form", etc. for Role.
        match self:
            case Role.LIST:
                return {"template_name_suffix": "_list"}
            case Role.DETAIL:
                return {"template_name_suffix": "_detail"}
            case Role.CREATE | Role.UPDATE:
                return {"template_name_suffix": "_form"}
            case Role.DELETE:
                return {"template_name_suffix": "_confirm_delete"}

    @property
    def url_name_component(self):
        return self.value

    def url_pattern(self, view_cls):
        url_base = view_cls.url_base
        url_kwarg = view_cls.lookup_url_kwarg or view_cls.lookup_field
        path_converter = view_cls.path_converter
        match self:
            case Role.LIST:
                return f"{url_base}/"
            case Role.DETAIL:
                return f"{url_base}/<{path_converter}:{url_kwarg}>/"
            case Role.CREATE:
                return f"{url_base}/new/"
            case Role.UPDATE:
                return f"{url_base}/<{path_converter}:{url_kwarg}>/edit/"
            case Role.DELETE:
                return f"{url_base}/<{path_converter}:{url_kwarg}>/delete/"

    def get_url(self, view_cls):
        return path(
            self.url_pattern(view_cls),
            view_cls.as_view(role=self),
            name=f"{view_cls.url_base}-{self.url_name_component}",
        )

    def reverse(self, view, object=None):
        url_name = f"{view.url_base}-{self.url_name_component}"
        url_kwarg = view.lookup_url_kwarg or view.lookup_field
        match self:
            case Role.LIST | Role.CREATE:
                return reverse(url_name)
            case _:
                return reverse(
                    url_name,
                    kwargs={url_kwarg: getattr(object, view.lookup_field)},
                )


ROLE_LIST = list[Role]


class CRUDView(View):
    """
    CRUDView is Neapolitan's core. It provides the standard list, detail,
    create, edit, and delete views for a model, as well as the hooks you need to
    be able to customise any part of that.
    """

    role: Role | None = None
    model: models.Model | None = None
    fields: list[str] | None = None  # TODO: handle this being None.

    # Object lookup parameters. These are used in the URL kwargs, and when
    # performing the model instance lookup.
    # Note that if unset then `lookup_url_kwarg` defaults to using the same
    # value as `lookup_field`.
    lookup_field = "pk"
    lookup_url_kwarg = None
    path_converter = "int"
    object = None

    # All the following are optional, and fall back to default values
    # based on the 'model' shortcut.
    # Each of these has a corresponding `.get_<attribute>()` method.
    queryset = None
    form_class = None
    template_name = None
    context_object_name = None

    # Pagination parameters.
    # Set `paginate_by` to an integer value to turn pagination on.
    paginate_by = None
    page_kwarg = "page"
    allow_empty = True

    # Suffix that should be appended to automatically generated template names.
    template_name_suffix = None

    def list(self, request, *args, **kwargs):
        """GET handler for the list view."""

        queryset = self.get_queryset()
        filterset = self.get_filterset(queryset)
        if filterset is not None:
            queryset = filterset.qs

        if not self.allow_empty and not queryset.exists():
            raise Http404

        paginate_by = self.get_paginate_by()
        if paginate_by is None:
            # Unpaginated response
            self.object_list = queryset
            context = self.get_context_data(
                page_obj=None,
                is_paginated=False,
                paginator=None,
                filterset=filterset,
            )
        else:
            # Paginated response
            page = self.paginate_queryset(queryset, paginate_by)
            self.object_list = page.object_list
            context = self.get_context_data(
                page_obj=page,
                is_paginated=page.has_other_pages(),
                paginator=page.paginator,
                filterset=filterset,
            )

        return self.render_to_response(context)

    def detail(self, request, *args, **kwargs):
        """GET handler for the detail view."""

        self.object = self.get_object()
        context = self.get_context_data()
        return self.render_to_response(context)

    def show_form(self, request, *args, **kwargs):
        """GET handler for the create and update form views."""

        if self.role is Role.UPDATE:
            self.object = self.get_object()
        form = self.get_form(instance=self.object)
        context = self.get_context_data(form=form)
        return self.render_to_response(context)

    def process_form(self, request, *args, **kwargs):
        """POST handler for the create and update form views."""

        if self.role is Role.UPDATE:
            self.object = self.get_object()
        form = self.get_form(
            data=request.POST,
            files=request.FILES,
            instance=self.object,
        )
        if form.is_valid():
            return self.form_valid(form)
        return self.form_invalid(form)

    def confirm_delete(self, request, *args, **kwargs):
        """GET handler for the delete confirmation view."""

        self.object = self.get_object()
        context = self.get_context_data()
        return self.render_to_response(context)

    def process_deletion(self, request, *args, **kwargs):
        """POST handler for the delete confirmation view."""

        self.object = self.get_object()
        self.object.delete()
        return HttpResponseRedirect(self.get_success_url())

    # Queryset and object lookup

    def get_queryset(self):
        """
        Returns the base queryset for the view.

        Either used as a list of objects to display, or as the queryset
        from which to perform the individual object lookup.
        """
        if self.queryset is not None:
            return self.queryset._clone()

        if self.model is not None:
            return self.model._default_manager.all()

        msg = (
            "'%s' must either define 'queryset' or 'model', or override "
            + "'get_queryset()'"
        )
        raise ImproperlyConfigured(msg % self.__class__.__name__)

    def get_object(self):
        """
        Returns the object the view is displaying.
        """
        queryset = self.get_queryset()
        lookup_url_kwarg = self.lookup_url_kwarg or self.lookup_field

        try:
            lookup = {self.lookup_field: self.kwargs[lookup_url_kwarg]}
        except KeyError:
            msg = "Lookup field '%s' was not provided in view kwargs to '%s'"
            raise ImproperlyConfigured(
                msg % (lookup_url_kwarg, self.__class__.__name__)
            )

        return get_object_or_404(queryset, **lookup)

    # Form handling

    def get_form_class(self):
        """
        Returns the form class to use in this view.
        """
        if self.form_class is not None:
            return self.form_class

        if self.model is not None and self.fields is not None:
            return model_forms.modelform_factory(self.model, fields=self.fields)

        msg = (
            "'%s' must either define 'form_class' or both 'model' and "
            "'fields', or override 'get_form_class()'"
        )
        raise ImproperlyConfigured(msg % self.__class__.__name__)

    def get_form(self, data=None, files=None, **kwargs):
        """
        Returns a form instance.
        """
        cls = self.get_form_class()
        return cls(data=data, files=files, **kwargs)

    def form_valid(self, form):
        self.object = form.save()
        return HttpResponseRedirect(self.get_success_url())

    def form_invalid(self, form):
        context = self.get_context_data(form=form)
        return self.render_to_response(context)

    def get_success_url(self):
        assert self.model is not None, (
            "'%s' must define 'model' or override 'get_success_url()'"
            % self.__class__.__name__
        )
        if self.role is Role.DELETE:
            success_url = reverse(f"{self.url_base}-list")
        else:
            success_url = reverse(
                f"{self.url_base}-detail", kwargs={"pk": self.object.pk}
            )
        return success_url

    # Pagination and filtering

    def get_paginate_by(self):
        """
        Returns the size of pages to use with pagination.
        """
        return self.paginate_by

    def get_paginator(self, queryset, page_size):
        """
        Returns a paginator instance.
        """
        return Paginator(queryset, page_size)

    def paginate_queryset(self, queryset, page_size):
        """
        Paginates a queryset, and returns a page object.
        """
        paginator = self.get_paginator(queryset, page_size)
        page_kwarg = self.kwargs.get(self.page_kwarg)
        page_query_param = self.request.GET.get(self.page_kwarg)
        page_number = page_kwarg or page_query_param or 1
        try:
            page_number = int(page_number)
        except ValueError:
            if page_number == "last":
                page_number = paginator.num_pages
            else:
                msg = "Page is not 'last', nor can it be converted to an int."
                raise Http404(_(msg))

        try:
            return paginator.page(page_number)
        except InvalidPage as exc:
            msg = "Invalid page (%s): %s"
            raise Http404(_(msg) % (page_number, str(exc)))

    def get_filterset(self, queryset=None):
        filterset_class = getattr(self, "filterset_class", None)
        filterset_fields = getattr(self, "filterset_fields", None)

        if filterset_class is None and filterset_fields:
            filterset_class = filterset_factory(self.model, fields=filterset_fields)

        if filterset_class is None:
            return None

        return filterset_class(
            self.request.GET,
            queryset=queryset,
            request=self.request,
        )

    # Response rendering

    def get_context_object_name(self, is_list=False):
        """
        Returns a descriptive name to use in the context in addition to the
        default 'object'/'object_list'.
        """
        if self.context_object_name is not None:
            return self.context_object_name

        elif self.model is not None:
            fmt = "%s_list" if is_list else "%s"
            return fmt % self.model._meta.object_name.lower()

        return None

    def get_context_data(self, **kwargs):
        kwargs["view"] = self
        kwargs["object_verbose_name"] = self.model._meta.verbose_name
        kwargs["object_verbose_name_plural"] = self.model._meta.verbose_name_plural
        kwargs["create_view_url"] = reverse(f"{self.url_base}-create")

        if getattr(self, "object", None) is not None:
            kwargs["object"] = self.object
            context_object_name = self.get_context_object_name()
            if context_object_name:
                kwargs[context_object_name] = self.object

        if getattr(self, "object_list", None) is not None:
            kwargs["object_list"] = self.object_list
            context_object_name = self.get_context_object_name(is_list=True)
            if context_object_name:
                kwargs[context_object_name] = self.object_list

        return kwargs

    def get_template_names(self):
        """
        Returns a list of template names to use when rendering the response.

        If `.template_name` is not specified, uses the
        "{app_label}/{model_name}{template_name_suffix}.html" model template
        pattern, with the fallback to the
        "neapolitan/object{template_name_suffix}.html" default templates.
        """
        if self.template_name is not None:
            return [self.template_name]

        if self.model is not None and self.template_name_suffix is not None:
            return [
                f"{self.model._meta.app_label}/"
                f"{self.model._meta.object_name.lower()}"
                f"{self.template_name_suffix}.html",
                f"neapolitan/object{self.template_name_suffix}.html",
            ]
        msg = (
            "'%s' must either define 'template_name' or 'model' and "
            "'template_name_suffix', or override 'get_template_names()'"
        )
        raise ImproperlyConfigured(msg % self.__class__.__name__)

    def render_to_response(self, context):
        """
        Given a context dictionary, returns an HTTP response.
        """
        return TemplateResponse(
            request=self.request, template=self.get_template_names(), context=context
        )

    # URLs and view callables

    @classonlymethod
    def as_view(cls, role: Role, **initkwargs):  # type: ignore[override]
        """Main entry point for a request-response process."""
        for key in initkwargs:
            if key in cls.http_method_names:
                raise TypeError(
                    "The method name %s is not accepted as a keyword argument "
                    "to %s()." % (key, cls.__name__)
                )
            if key in [
                "list",
                "detail",
                "show_form",
                "process_form",
                "confirm_delete",
                "process_deletion",
            ]:
                raise TypeError(
                    "CRUDView handler name %s is not accepted as a keyword argument "
                    "to %s()." % (key, cls.__name__)
                )
            if not hasattr(cls, key):
                raise TypeError(
                    "%s() received an invalid keyword %r. as_view "
                    "only accepts arguments that are already "
                    "attributes of the class." % (cls.__name__, key)
                )

        def view(request, *args, **kwargs):
            self = cls(**initkwargs, **role.extra_initkwargs())
            self.role = role
            self.setup(request, *args, **kwargs)
            if not hasattr(self, "request"):
                raise AttributeError(
                    f"{cls.__name__} instance has no 'request' attribute. Did you "
                    "override setup() and forget to call super()?"
                )

            for method, action in role.handlers().items():
                handler = getattr(self, action)
                setattr(self, method, handler)

            return self.dispatch(request, *args, **kwargs)

        view.view_class = cls  # type: ignore[attr-defined]
        view.view_initkwargs = initkwargs  # type: ignore[attr-defined]

        # __name__ and __qualname__ are intentionally left unchanged as
        # view_class should be used to robustly determine the name of the view
        # instead.
        view.__doc__ = cls.__doc__
        view.__module__ = cls.__module__
        view.__annotations__ = cls.dispatch.__annotations__
        # Copy possible attributes set by decorators, e.g. @csrf_exempt, from
        # the dispatch method.
        view.__dict__.update(cls.dispatch.__dict__)

        # Mark the callback if the view class is async.
        # if cls.view_is_async:
        #     markcoroutinefunction(view)

        return view

    @classproperty
    def url_base(cls):
        """
        The base component of generated URLs.

        Defaults to the model's name, but may be overridden by setting
        `url_base` directly on the class, overriding this property::

            class AlternateCRUDView(CRUDView):
                model = Bookmark
                url_base = "something-else"
        """
        return cls.model._meta.model_name

    @classonlymethod
    def get_urls(
        cls,
        roles: ROLE_LIST | None = None,
        *,
        exclude: ROLE_LIST | None = None,
    ):
        """Classmethod to generate URL patterns for the view."""
        if roles is not None and exclude is not None:
            msg = "Cannot specify both roles and exclude."
            raise ValueError(msg)
        if exclude is not None:
            roles = [role for role in Role if role not in exclude]
        if roles is None:
            roles = [role for role in Role]
        return [role.get_url(cls) for role in roles]


# region toppings
class ModelMeta(BaseModel):
    app_label: str
    name: str


class ViewAction(BaseModel):
    name: str
    text: str
    icon: str | Icon | None = None
    button_class: str = "btn btn-primary"
    url: str
    permissions_required: str | list[str] | None = None
    requires_lookup: bool = False
    strict_url_name: bool = False

    def check_permissions(self, view):
        user = view.request.user
        if isinstance(self.permissions_required, list):
            is_authorized = user.has_perms(self.permissions_required)
        else:
            is_authorized = user.has_perm(self.permissions_required)
        if not is_authorized:
            return redirect(f"{settings.LOGIN_URL}?next{view.request.path}")
        return True

    def icon_html(self):
        if self.icon is None:
            return None
        if isinstance(self.icon, str):
            return self.icon
        if isinstance(self.icon, Icon):
            return self.icon.html
        return None

    def reverse(self, view: type[CRUDView]):
        if self.strict_url_name:
            return reverse_lazy(self.url)

        model_meta = self.model_meta(view)
        app_label = model_meta.app_label
        model_name = model_meta.name
        model_name = model_name if model_name is not None else ""

        try:
            url = self.url.format(app_label=app_label, model_name=model_name)
        except KeyError:
            url = self.url
        url_formatted = url != self.url

        if not url_formatted:
            if model_name not in url:
                url = f"{view.url_base}-{self.url}"

        if self.requires_lookup:
            msg = f"action name: {self.name}"
            logger.debug(msg)

            lookup_arg = getattr(view.object, view.lookup_field)
            url_old = url
            url = url.format(**view.object.__dict__)
            if url != url_old:
                return url

            msg = f"{self.__class__.__name__} lookup url name: {url}"
            logger.debug(msg)

            return reverse_lazy(url, args=[lookup_arg])

        return reverse_lazy(url)

    # TODO: figure out why this gets a "NoReverseMatch"
    def reverse_model(self, *, model: type[models.Model], view_class: type[View]):
        warnings.warn(
            "this doesn't work in the real world, not sure why",
            UserWarning,
            stacklevel=2,
        )
        # model_meta = self.get_model_meta(model)
        url_pattern_name = Role.DETAIL.get_url(view_class)
        msg = f"{self.__class__.__name__} url pattern name: {url_pattern_name}"
        logger.debug(msg)
        lookup_field = (
            "pk" if not hasattr(view_class, "lookup_field") else view_class.lookup_field
        )
        lookup_arg = getattr(model, lookup_field)

        try:
            return reverse(url_pattern_name, args=[lookup_arg])
        except NoReverseMatch:
            msg = "reverse_model doesn't work, do this manually"
            logger.warning(msg)
            return None

    def model_meta(self, view):
        return ModelMeta(
            app_label=view.model._meta.app_label,  # noqa: SLF001
            name=get_model_name(view.model),
        )

    def get_model_meta(self, model: type[models.Model]):
        return ModelMeta(
            app_label=model._meta.app_label,  # noqa: SLF001
            name=get_model_name(model),
        )

    def context_model_dump(
        self, view: type[CRUDView] | None = None, *, url: str | None = None
    ):
        simple_dump = self.model_dump(include={"name", "text", "icon", "button_class"})
        if isinstance(self.icon, Icon):
            simple_dump["icon"] = (
                self.icon.snippet if self.icon.has_snippet else self.icon.svg
            )
        if view is not None:
            simple_dump["url"] = self.reverse(view=view)
        elif url is not None:
            simple_dump["url"] = url
        return simple_dump


class BaseActions(Enum):
    DETAIL = ViewAction(
        name="detail",
        text="detail",
        icon=LibraryIcon.DETAIL,
        url="detail",
        requires_lookup=True,
    )
    UPDATE = ViewAction(
        name="update",
        text="edit",
        icon=LibraryIcon.UPDATE,
        url="update",
        requires_lookup=True,
    )
    LIST = ViewAction(
        name="list",
        text="list",
        button_class="btn btn-info",
        icon=LibraryIcon.LIST,
        url="list",
        requires_lookup=False,
    )
    CREATE = ViewAction(
        name="create",
        text="new",
        icon=LibraryIcon.CREATE,
        url="create",
        requires_lookup=False,
    )
    DELETE = ViewAction(
        name="delete",
        text="delete",
        icon=LibraryIcon.DELETE,
        url="delete",
        button_class="btn btn-danger",
        requires_lookup=True,
    )
    ADMIN = ViewAction(
        name="admin",
        text="admin",
        icon=LibraryIcon.ADMIN,
        url="admin:{app_label}_{model_name}_changelist",
        requires_lookup=False,
    )

    @classmethod
    def all(cls):
        return [e.value for e in cls]

    @classmethod
    def filter(cls, names: str | list[str]):
        """Return only views whose name is in `names`."""
        names = [names] if isinstance(names, str) else names

        return [action for action in cls if action.value.name in names]

    @classmethod
    def filter_negative(cls, names: str | list[str]):
        """Return only views whose name is NOT in `names`."""
        names = [names] if isinstance(names, str) else names

        return [action for action in cls if action.value.name not in names]


class ViewActions(BaseModel):
    name: str
    actions: list[ViewAction] = []

    def append(self, action: ViewAction):
        self.actions.append(action)

    def prepend(self, action: ViewAction):
        self.actions = [action, *self.actions]

    def inject_before_delete(self, action: ViewAction):
        actions = self.actions
        delete_action = None
        for idx, internal in enumerate(actions):
            if internal == BaseActions.DELETE.value:
                delete_action = actions.pop(idx)

        actions.append(action)

        if delete_action is not None:
            actions.append(delete_action)

        self.actions = actions

    def inject(self, action: ViewAction, index: int):
        self.actions.insert(index, action)


class ViewActionMixin:
    actions: (
        list[ViewAction | tuple[str, ViewAction] | tuple[str, list[ViewAction]]] | None
    ) = None
    role: Role | None = None
    exclude_actions: list[str] = []

    def get_actions_by_role(self) -> list[ViewAction] | None:
        msg = f"{self.__class__.__name__} actions: {self.actions}"
        logger.debug(msg)
        if self.actions is None:
            logger.debug(msg)

            return None
        if self.role is None:
            return None
        if not isinstance(self.actions, list):
            msg = f"actions must be a list, not {type(self.actions)}"
            raise TypeError(msg)
        actions_list = []
        for action in self.actions:
            if isinstance(action, ViewAction):
                msg = f"generic ViewAction: {action}"
                logger.debug(msg)
                actions_list.append(action)
            if isinstance(action, tuple):
                role_name, view_actions = action
                if Role(role_name) == self.role:
                    if isinstance(view_actions, list):
                        actions_list.extend(view_actions)
                    else:
                        actions_list.append(view_actions)

        msg = f"actions_list: {actions_list}"
        logger.debug(msg)
        return actions_list

    class ViewActionMixinEnum(Enum):
        LIST = ViewActions(name="list", actions=[BaseActions.CREATE.value])
        HISTORY = ViewActions(name="list")
        CREATE = ViewActions(name="create")
        UPDATE = ViewActions(name="update")
        DETAIL = ViewActions(
            name="detail",
            actions=[
                BaseActions.UPDATE.value,
                BaseActions.LIST.value,
                # AdminActionsEnum.CHANGE.value,
                # AdminActionsEnum.HISTORY.value,
                BaseActions.DELETE.value,
            ],
        )
        DELETE = ViewActions(name="delete")

        @classmethod
        def by_name(cls, name: str):
            for enum in cls:
                if enum.value.name == name:
                    return enum

            return None

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)  # type: ignore[misc]
        msg = "in get_context_data"
        logger.debug(msg)
        context["actions"] = self.get_action_objects()

        return context

    def get_action_objects(self):
        msg = "in get_action_objects"
        logger.debug(msg)

        if self.role is None:
            return None

        role_view_actions = self.ViewActionMixinEnum.by_name(self.role.value).value

        if role_view_actions is None:
            return None

        if len(self.exclude_actions) != 0:
            msg = f"actions excluded: {self.exclude_actions}"
            logger.debug(msg)

            msg = (
                f"action names: {[action.name for action in role_view_actions.actions]}"
            )
            logger.debug(msg)

            role_view_actions.actions = [
                action
                for action in role_view_actions.actions
                if action.name not in self.exclude_actions
            ]

        view_actions = ViewActions(
            name=f"{self.__class__.__name__}_{self.role.value}",
            actions=list(role_view_actions.actions),
        )

        custom_actions = self.get_actions_by_role()

        if custom_actions is not None:
            assert isinstance(
                custom_actions, list
            ), f"custom_actions must be a list, not {type(custom_actions)}"
            for action in custom_actions:
                assert isinstance(
                    action, ViewAction
                ), f"{action} must be of type ViewAction"
                msg = f"custom action: {action}"
                logger.debug(msg)
                view_actions.inject_before_delete(action)

        return [
            action.context_model_dump(self)  # type: ignore[arg-type]
            for action in view_actions.actions
            if action.check_permissions(self)
        ]

    def get_actions(self):
        """Create a dictionary of actions to be used as buttons.

        Primarily intended for detail views, but may be adapted later
        """
        actions = self.get_action_objects()
        instance_object = getattr(self, "object", None)
        if instance_object is None:
            actions = [
                action["action"]
                for action in actions
                if not action["action"].requires_lookup
            ]

        return [action.context_model_dump(self) for action in actions]


class InlineView:
    """Create an inline for related objects on a model.

    If `table_class` includes fields, use `exclude` to remove the related field;
    `fields` will only be used to add additional fields, if possible

    Arguments:
      label: label to be used in template
      model: the model the the table should reference
      fields: fields to include in table; will not exclude if `table_class` has fields
      exclude: fields to remove from existing `table_class`
      table_class: base model class to use
      related_field_name: name of field on parent model that links to related model
    """

    # TODO: automatically determine `related_field_name` and exclude related column
    label: str | None = None
    model: type[models.Model]
    fields: list[str] = []
    exclude: list[str] | None = None
    table_class: type[Table] = Table
    related_field_name: str
    order_by: str | None = None

    @classmethod
    def get_table(cls, view: "ObjectView") -> tuple[str, type[Table]]:
        cls.label = cls.label or get_model_verbose_name_plural(cls.model)
        query_kwargs = {cls.related_field_name: view.get_object()}
        data = cls.model._default_manager.filter(**query_kwargs)  # noqa: SLF001
        table = table_factory(
            cls.model, table=cls.table_class, fields=cls.fields, exclude=cls.exclude
        )
        table_kwargs: dict[str, Any] = {"orderable": False}
        if cls.order_by is not None:
            msg = f"order_by: {cls.order_by}"
            logger.debug(msg)

            table_kwargs["order_by"] = cls.order_by
        return cls.label, table(data, **table_kwargs)


class ObjectView(ViewActionMixin, CRUDView):
    list_view: type[View] | None = None
    update_view: type[View] | None = None
    create_view: type[View] | None = None
    detail_view: type[View] | None = None
    delete_view: type[View] | None = None
    widgets: dict[str, type[Widget] | Widget] | None = None
    readonly_fields: list[str] = []
    inlines: "Iterable[type[InlineView]] | None" = None

    @classonlymethod
    def get_view_class(cls, role: Role) -> type[View]:  # noqa: N805
        match role:
            case Role.LIST:
                view = cls.list_view
            case Role.UPDATE:
                view = cls.update_view
            case Role.CREATE:
                view = cls.create_view
            case Role.DETAIL:
                view = cls.detail_view
            case Role.DELETE:
                view = cls.delete_view
            case _:
                msg = f"get_view_class requires a valid role: {role}"
                raise ValueError(msg)

        return view or cls

    @override
    def detail(self, request, *args, **kwargs):
        """GET handler for the detail view.

        Adds collection of fields with their labels and values
        Adds `update_view_url` for "edit" button
        """
        msg = f"template_name: {self.get_template_names()}"
        logger.debug(msg)

        self.object = self.get_object()
        context = self.get_context_data()

        fields_list = get_model_fields(self.object, fields=self.fields)

        try:
            TaggableManager = importlib.import_module("taggit.managers").TaggableManager
        except ModuleNotFoundError:
            TaggableManager = None

        fields_dict = []
        for field in fields_list:
            field_type = "unknown"
            field_value = getattr(self.object, field.name)

            if TaggableManager is not None:
                if isinstance(field, TaggableManager):
                    widget = DetailWidget(
                        field_class=TaggableManager, template_name="tag_field.html"
                    )
                    field_value = widget.render_to_string(field_value.all())
                    field_type = "taggit"

            if isinstance(field, ManyToManyField):
                widget = DetailWidget(
                    field_class=ManyToManyField, template_name="m2m_field.html"
                )
                field_value = widget.render_to_string(field_value.all())
                field_type = "m2m"

            if field.name == "assigned":
                msg = f"type assigned: {type(field)}"
                logger.debug(msg)

            if isinstance(field, models.CharField):
                field_value = gettext(field_value)
            if isinstance(field_value, bool):
                field_type = "bool"
            if isinstance(field_value, str):
                field_type = "str"

            fields_dict.append(
                {
                    "verbose_name": field.verbose_name,
                    "value": field_value,
                    "help_text": field.help_text,
                    "field_type": field_type,
                }
            )
        context["fields_dict"] = fields_dict

        if (inlines := self.inlines) is not None:
            context["inlines"] = [inline.get_table(self) for inline in inlines]

        return self.render_to_response(context)

    @override
    def get_template_names(self):
        """Returns a list of template names to use when rendering the response.

        Overridden to add a local generic template location

        If `.template_name` is not specified, uses the
        "{app_label}/{model_name}{template_name_suffix}.html" model template
        pattern, with the fallback to the
        "neapolitan/object{template_name_suffix}.html" default templates.
        """
        if self.template_name is not None:
            return [self.template_name]

        if self.model is not None and self.template_name_suffix is not None:
            return [
                f"{self.model._meta.app_label}/"  # noqa: SLF001
                f"{self.model._meta.object_name.lower()}"  # noqa: SLF001
                f"{self.template_name_suffix}.html",
                f"object{self.template_name_suffix}.html",
                f"neapolitan/object{self.template_name_suffix}.html",
            ]
        msg = (
            "'%s' must either define 'template_name' or 'model' and "
            "'template_name_suffix', or override 'get_template_names()'"
        )
        raise ImproperlyConfigured(msg % self.__class__.__name__)

    @override
    @classonlymethod
    def get_urls(
        cls,  # noqa: N805
        roles: list[Role] | None = None,
        *,
        exclude: list[Role] | None = None,
    ):
        """Classmethod to generate URL patterns for the view."""
        roles = list(Role) if roles is None else roles

        msg = f"`get_url` roles: {[role.name for role in roles]}"
        logger.debug(msg)

        view_tuples = [(role, cls.get_view_class(role)) for role in roles]

        if len(view_tuples) > 0:
            role, view_class = view_tuples[0]

            msg = f"first url: {role.get_url(view_class)}"
            logger.debug(msg)

        return [role.get_url(view_class) for role, view_class in view_tuples]

    @override
    def get_form_class(self):
        if self.form_class is not None:
            return self.form_class

        if self.model is not None and self.fields is not None:
            fields = [
                field for field in self.fields if field not in self.readonly_fields
            ]
            return modelform_factory(
                self.model,
                CrispyModelForm,
                fields=fields,
                widgets=self.widgets,
            )

        msg = (
            "'%s' must either define 'form_class' or both 'model' and "
            "'fields', or override 'get_form_class()'"
        )
        raise ImproperlyConfigured(msg % self.__class__.__name__)

    def get_context_data(self, **kwargs: Any):
        context = super().get_context_data(**kwargs)
        assert (
            self.role is not None
        ), f"{self.__class__.__name__} must have a non-None role"

        if self.role in [Role.CREATE, Role.UPDATE]:
            current_object = getattr(self, "object", None)
            form = context["form"]
            form.helper.form_id = (
                f"id-{get_model_name(self.model)}"
                f"{self.role.url_name_component.capitalize()}Form"
            )
            form.helper.form_action = self.role.reverse(self, current_object)

            form.helper.add_input(Submit("submit", "Submit"))

            context["form"] = form

            msg = f"Form helper action: {form.helper.form_action}"
            logger.debug(msg)

        return context

    @override
    def get_success_url(self):
        if self.model is None and self.form_class._meta.model is None:  # noqa: SLF001
            msg = (
                f"'{self.__class__.__name__}' must define 'model' or "
                "override 'get_success_url()'"
            )
            raise ValueError(msg)

        if self.model is not None:
            model = self.model
        elif self.form_class is not None:
            model = self.form_class._meta.model  # noqa: SLF001

        if self.role == Role.DELETE:
            return Role.LIST.reverse(self)

        if self.object is not None:
            current_object = model.objects.get(pk=self.object.pk)

        url = Role.DETAIL.reverse(self, current_object)

        msg = f"{self.__class__.__name__} success url: {url}"
        logger.debug(msg)

        return url

    def model_meta(self):
        return ModelMeta(
            app_label=self.model._meta.app_label,  # noqa: SLF001
            name=get_model_name(self.model),
        )

    def get_view_context(self):
        project_role_list = [Role.LIST, Role.CREATE]
        view_classes = {
            role.value: self.get_view_class(role=role) for role in project_role_list
        }

        model_meta = self.model_meta()

        model_views = [
            (model_meta.object_name, view_class) for view_class in view_classes
        ]

        return {model_meta.app_label: model_views}


# endregion toppings
