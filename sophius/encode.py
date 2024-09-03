from sophius.templates import *
import numpy as np

TEMPLATES = (
    LinearTmpl(),
    BatchNorm2dTmpl(),
    ReLUTmpl(),
    LeakyReLUTmpl(),
    PReLUTmpl(),
    DropoutTmpl(),
    Dropout2dTmpl(),
    FlattenTmpl(),
    Conv2dTmpl(),
    MaxPool2dTmpl(),
    AvgPool2dTmpl(),
    GlobalAvgPool2dTmpl(),
)

ENCODING_SIZE = 32

class Encoder:
    def __init__(self, templates=TEMPLATES, size=ENCODING_SIZE):
        self.templates = templates
        self.types = [t.type for t in templates]
        self.size = size

    def encode_template(self, template):
        # One-hot encoding for the template type
        type_encoding = np.zeros(len(self.types), dtype=np.uint32)
        idx = self.types.index(template.type)
        type_encoding[idx] = 1

        param_encoding = []

        for param_name, param_data in template.config_data.items():
            # skip non-learnable params
            if not template.config[param_name].learnable:
                continue

            value = template.config[param_name].value
            param_range = param_data.get('range')

            # Handle discrete ranges (categorical values)
            if isinstance(param_range, list):
                encoding = np.zeros(len(param_range), dtype=np.uint32)
                if value in param_range:
                    idx = param_range.index(value)
                    encoding[idx] = 1
                else:
                    raise ValueError(f"[{template.name}] Value {value} not in the provided range for {param_name}")

            else:
                raise ValueError(f"[{template.name}] Unsupported parameter type for {param_name}: {value}")

            param_encoding.append(encoding)

            # Concatenate all encodings into a single vector for the template
        final_encoding = np.concatenate([type_encoding] + param_encoding)
        final_encoding = np.pad(final_encoding, (0, max(0, self.size - len(final_encoding))))

        return final_encoding

    def encode_model(self, model):
        res = []
        for t in model.get_templates()[:-1]:
            res.append(self.encode_template(t))
        return np.array(res)
    def decode(self, encoding):
        pass
