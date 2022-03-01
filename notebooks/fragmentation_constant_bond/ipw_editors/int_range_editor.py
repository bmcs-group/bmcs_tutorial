
import traits.api as tr
from bmcs_utils.editors.editor_factory import EditorFactory
import ipywidgets as ipw

class IntRangeEditor(EditorFactory):
    low = tr.Int(0)
    high = tr.Int(1)
    low_name = tr.Str
    high_name = tr.Str
    continuous_update = tr.Bool(False)

    def render(self):
        if self.low_name:
            self.low = getattr(self.model, str(self.low_name))
            self.model.observe(self.reset_low, 'state_changed')
        if self.high_name:
            self.high = getattr(self.model, str(self.high_name))
            self.model.observe(self.reset_high,'state_changed')
        self.ipw_editor = ipw.IntSlider(
            description=self.label,
            value=self.value, min=self.low, max=self.high,
            tooltip=self.tooltip,
            disabled=self.disabled
        )
        return self.ipw_editor

    def reset_low(self, event):
        self.ipw_editor.max = getattr(self.model, str(self.low_name))

    def reset_high(self, event):
        self.ipw_editor.max = getattr(self.model, str(self.high_name))

