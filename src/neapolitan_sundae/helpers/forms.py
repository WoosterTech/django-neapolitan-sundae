from crispy_forms.helper import FormHelper  # type: ignore[import-untyped]
from django.forms import ModelForm


class CrispyModelForm(ModelForm):
    """Use only as a base form class for factories."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_class = "blueForms"
        self.helper.form_method = "post"
