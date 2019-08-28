from applicationinsights import TelemetryClient
from uuid import uuid4
import sys

instrumentation_key = "7908c1fe-8c00-46a6-ae66-645d6a14db06"

class ScriptInstrumentor(object):
    def __init__(self, event_context):
        self.tc = TelemetryClient(instrumentation_key)
        self.tc.context.session.id = str(uuid4())
        self.event_context = event_context
        self.event_context["epoch"] = -1

    def set_context(self, key, value):
        self.event_context[key] = value

    def start_epoch(self, epoch):
        self.event_context["epoch"] = epoch
        self.tc.track_event("epoch_start", self.event_context)

    def end_epoch(self):
        self.tc.track_event("epoch_end", self.event_context)
        self.tc.flush()

    def start_dataset_load(self):
        self.tc.track_event("dataset_load_start", self.event_context)

    def end_dataset_load(self):
        self.tc.track_event("dataset_load_end", self.event_context)

    def start_dataset_split(self):
        self.tc.track_event("dataset_split_start", self.event_context)

    def end_dataset_split(self):
        self.tc.track_event("dataset_split_end", self.event_context)

    def start_phase(self, phase):
        self.tc.track_event("{}_start".format(phase), self.event_context)

    def end_phase(self, phase, loss):
        self.tc.track_event("{}_end".format(phase), self.event_context, {"loss": loss})

    def trace_memory(self, memory_metrics):
        self.tc.track_event("trace_memory", self.event_context, memory_metrics)

    def log_exception(self, additional_context):
        self.tc.track_exception(*sys.exc_info(), properties=({**self.event_context, **additional_context}))
        self.tc.flush()

    def log_bad_image(self, image_name, min_value, max_value, tensor_name):
        self.tc.track_event("bad_image", self.event_context, {"min": min_value, "max": max_value, "name": image_name, "tensor": tensor_name})