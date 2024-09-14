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
    """
    Converts 2D array of bit vectors to hex representation,
    @param arr: 2D numpy bit array, ex [[0, 1, 0, 1], [1, 0, 1, 1]]
    @return: str hex representation, ex '050B'
    """

    def bits_to_hex(vector):
        bit_string = ''.join(vector.astype(str))
        return hex(int(bit_string, 2))[2:].zfill(8)

    # Flatten the 2D array and convert each 32-bit vector to hex
    return ''.join(bits_to_hex(row) for row in arr)


def str_to_vec(hex_str: str) -> np.ndarray:
    """
    Converts hex representation of 2D array of bit vectors to bit vector,
    @param hex_str: hex representation, ex '050B'
    @return: 2D numpy bit array, ex [[0, 1, 0, 1], [1, 0, 1, 1]]
    """
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
    """
    Handles encoding of module templates to bit vectors.
    """
    def __init__(self, templates=TEMPLATES, size=ENCODING_SIZE):
        self.templates = templates
        self.types = [t.__name__ for t in templates]
        self.size = size

    def encode_template(self, template: ModuleTemplate_, dtype=np.uint8):
        """
        Encodes template to bit vector.
        First 12 bytes is onehot encoded template type, taken from TEMPLATES const.
        the rest of 20 bytes are encoded params from the module config_data.
        @param template: ModuleTmpl instance to encode, ex Conv2d
        @param dtype: bit vector datatype
        @return: 1D bit vector encoding, ex [0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 1 0]
        """
        # One-hot encoding for the template type
        type_encoding = np.zeros(len(self.types), dtype=dtype)
        idx = self.types.index(template.type)
        type_encoding[idx] = 1

        param_encoding = []

        for param_name, param_data in template.config.items():
            value = template.params[param_name]
            param_range = param_data.get('range')

            # skip non-learnable params
            if param_range is None:
                continue

            # Handle discrete ranges (categorical values)
            if isinstance(param_range, list):
                encoding = np.zeros(len(param_range), dtype=dtype)
                if value in param_range:
                    idx = param_range.index(value)
                    encoding[idx] = 1
                else:
                    raise ValueError(f"[{template.torch_name}] Value {value} not in the provided range for {param_name}")

            else:
                raise ValueError(f"[{template.torch_name}] Unsupported parameter type for {param_name}: {value}")

            param_encoding.append(encoding)

            # Concatenate all encodings into a single vector for the template
        final_encoding = np.concatenate([type_encoding] + param_encoding)

        if len(final_encoding) > self.size:
            raise ValueError(f'Template encoding too long: {len(final_encoding)}, too many params?')

        final_encoding = np.pad(final_encoding, (0, max(0, self.size - len(final_encoding))))

        return final_encoding

    def model2vec(self, model: ModelTmpl) -> np.ndarray:
        """
        Converts model to 2D bit vector. Omits the last template, as it is always linear.
        @param model: ModelTmpl instance to encode
        @return: 2D bit vector encoding,
        Example:
        [[0, 1, 0, 1],
         [1, 0, 1, 1],
         [1, 0, 0, 1]]
        """
        res = []
        for t in model.templates[:-1]:
            res.append(self.encode_template(t))
        return np.array(res)

    def model2hash(self, model: ModelTmpl) -> str:
        """
        Converts model to hash string representation, by converting each bit vector
        to hex string and stacking them together.
        @param model: ModelTmpl instance to encode
        @return: hash string representation, ex '050B'
        """
        return vec_to_str(self.model2vec(model))

    def decode_template(self, input_vec) -> ModuleTemplate_:
        """
        Decodes a bit vector back to a template object.
        @param input_vec: 1D bit vector encoding
        @return: Decoded template object
        """
        # Decode the template type from the first 12 bits
        type_encoding = input_vec[:len(self.types)]
        idx = np.argmax(type_encoding)
        template_type = self.templates[idx]

        # Start decoding the parameters
        param_encoding_start = len(self.types)
        params = {}

        for name, data in template_type.config.items():
            param_range = data.get('range')

            if param_range is None:
                continue

            if isinstance(param_range, list):
                param_encoding = input_vec[param_encoding_start: param_encoding_start + len(param_range)]
                idx = np.argmax(param_encoding)
                param_value = param_range[idx]
                param_encoding_start += len(param_range)
            else:
                raise ValueError(f"Unsupported parameter type for {name}")

            params[name] = param_value

        # Create the template instance with decoded parameters
        template = template_type(**params)

        return template

    def hash2model(self, hash_str: str, in_shape=None, out_shape=None) -> ModelTmpl:
        templates = []

        arr = str_to_vec(hash_str)

        for vec in arr:
            templates.append(self.decode_template(vec))

        # last layer is always linear
        templates.append(LinearTmpl())

        return ModelTmpl(in_shape, out_shape, *templates)


