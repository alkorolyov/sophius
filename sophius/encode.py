import sophius
from sophius.templates import *
import numpy as np

TEMPLATES = (
    LinearTmpl,
    BatchNorm2dTmpl,
    ReLUTmpl,
    LeakyReLUTmpl,
    PReLUTmpl,
    DropoutTmpl,
    Dropout2dTmpl,
    FlattenTmpl,
    Conv2dTmpl,
    MaxPool2dTmpl,
    AvgPool2dTmpl,
    GlobalAvgPool2dTmpl,
)

ENCODING_SIZE = 32


def vec_to_str(arr: np.ndarray) -> str:
    def bits_to_hex(vector):
        bit_string = ''.join(vector.astype(str))
        return hex(int(bit_string, 2))[2:].zfill(8)

    # Flatten the 2D array and convert each 32-bit vector to hex
    return ''.join(bits_to_hex(row) for row in arr)


def str_to_vec(hex_str: str) -> np.ndarray:
    def hex_to_bits(hex_chunk):
        int_value = int(hex_chunk, 16)
        bit_string = bin(int_value)[2:].zfill(32)
        return np.array(list(bit_string), dtype=np.uint8)

    # Split the hex string into chunks of 8 characters (32 bits each)
    hex_chunks = [hex_str[i:i + 8] for i in range(0, len(hex_str), 8)]
    bit_rows = [hex_to_bits(chunk) for chunk in hex_chunks]

    # Convert the list of rows into a 2D NumPy array
    return np.array(bit_rows).reshape(-1, 32)


class Encoder:
    def __init__(self, templates=TEMPLATES, size=ENCODING_SIZE):
        self.templates = templates
        self.types = [t.__name__ for t in templates]
        self.size = size

    def encode_template(self, template, dtype=np.uint8):
        # One-hot encoding for the template type
        type_encoding = np.zeros(len(self.types), dtype=dtype)
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
                encoding = np.zeros(len(param_range), dtype=dtype)
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

    def encode_model(self, model) -> np.ndarray:
        res = []
        for t in model.get_templates()[:-1]:
            res.append(self.encode_template(t))
        return np.array(res)

    def encode_model_str(self, model) -> str:
        return vec_to_str(self.encode_model(model))

    def decode(self, encoding):
        pass
