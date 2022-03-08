from t1005_graph import *


class Header:
    """
    An agent for function interface.
    Each function name of option is guaranteed to be option.decorate(function_name).

    What is an header?
    Contain full information for cpp function head generation.
    Always capable for compatibility check.
    An optioned header

    may not need to consider lib dependency.
    may not need to consider namespace.
    """

    _option_head = "//=====[Option]====="
    _constant_output_format = "%.6f"

    def __init__(self, function_name: str,
                 supported_options: List[Option],
                 input_spec: List[str],
                 output_spec: List[str],
                 input_names: List[str],
                 output_names: List[str],
                 constant_derivative_channels: Dict[Tuple, float],
                 safe_check=True):
        self.function_name = function_name
        self.supported_options = supported_options
        self.input_spec = input_spec
        self.output_spec = output_spec
        self.input_names = input_names
        self.output_names = output_names

        if safe_check:
            self.safe_check()

        # the shared dict for all derivative orders.
        self.constant_derivative_channels = constant_derivative_channels

    def safe_check(self):
        """
        Check the validity of
        self.function_name
        self.input_spec
        self.output_spec
        self.input_names
        self.output_names
        :return:
        """
        assert is_valid_cpp_name(self.function_name), "Need valid function name."
        assert all([is_valid_cpp_name(item) for item in self.input_spec]), "Need valid input type."
        assert all([is_valid_lower_case_cpp_name(item) for item in self.input_names]), "Need valid input name."
        assert all([is_valid_cpp_name(item) for item in self.output_spec]), "Need valid output type."
        assert all([is_valid_lower_case_cpp_name(item) for item in self.output_names]), "Need valid output name."
        assert len(self.input_spec) == len(self.input_names)
        assert len(self.output_spec) == len(self.output_names)
        all_name_set = set(self.input_names)
        all_name_set.update(set(self.output_names))
        assert len(all_name_set) == len(self.input_names) + len(self.output_names), "Need no repeated names."

    def output_channels(self, option):
        """
        constant_derivative_channels must be valid
        list all output channels including zero order, skipping constant derivative ones.
        :param option:
        :return:
        """
        in_dim = len(self.input_spec)
        out_dim = len(self.output_spec)
        full_channels = full_output_channels_with_derivatives(in_dim, out_dim,
                                                              option.enable_1st_order_derivative(),
                                                              option.enable_2nd_order_derivative())
        channels = []
        for channel in full_channels:
            if channel not in self.constant_derivative_channels:
                channels.append(channel)

        return channels

    def find_missing_output_channels(self, option: Option, given_channels: Set[Tuple]):
        """
        a step to determine constant_derivative_channels.
        :param option:
        :param channels:
        :return:
        """
        in_dim = len(self.input_spec)
        out_dim = len(self.output_spec)
        full_channels = full_output_channels_with_derivatives(in_dim, out_dim,
                                                              option.enable_1st_order_derivative(),
                                                              option.enable_2nd_order_derivative())
        missing_channels = []
        for channel in full_channels:
            if channel not in given_channels:
                missing_channels.append(channel)

        return missing_channels

    def output_channel_name(self, channel: Tuple):
        assert len(channel) in {1, 2, 3}
        if len(channel) > 1:
            ch_names = [self.output_names[channel[0]]] + \
                       [self.input_names[i] for i in channel[1:]]
            return get_channel_name(tuple(ch_names))
        else:
            return self.output_names[channel[0]]

    def output_channel_type(self, channel: Tuple):
        assert len(channel) > 0
        return self.output_spec[channel[0]]

    def output_channel_from_name(self, name_string: str) -> Tuple:
        """
        Pre-requisite:
        self.input_names, self.output_names are ready
        :param name_string:
        :return:
        """
        # out
        # D2_out_D_in1_D_in2
        items = name_string.split("_")
        n = len(items)
        # 1 for 0 order, 4 for 1st order, 6 for 2nd order
        warning = "invalid field name:" + name_string
        assert n in {1, 4, 6}, warning

        def input_index_from_name(name: str):
            assert is_valid_lower_case_cpp_name(name), warning
            assert name in self.input_names, warning
            return self.input_names.index(name)

        def output_index_from_name(name: str):
            assert is_valid_lower_case_cpp_name(name), warning
            assert name in self.output_names, warning
            return self.output_names.index(name)

        for item in items:
            if is_valid_lower_case_cpp_name(item):
                assert (item in self.input_names) or (item in self.output_names), \
                    "invalid field name:" + name_string

        if n == 1:
            return output_index_from_name(items[0]),
        elif n == 4:
            return output_index_from_name(items[1]), input_index_from_name(items[3])
        elif n == 6:
            return output_index_from_name(items[1]), input_index_from_name(items[3]), input_index_from_name(items[5])

    def print_header_core(self):
        """
        to .h
        one line, one input/output
        /* xxx_name = 4.5*/ like this for constant derivative outputs
        you parse the xxx_name

        option must be headed.
        use raw option to fetch input spec. output specs.
        for constant derivative io, I bet you can make them inline in header.
        :return:
        """
        result = []

        for option in self.supported_options:
            # Print option lines
            option_lines = option.to_string().split('\n')
            result.append(self._option_head)
            for line in option_lines:
                result.append("//" + line)
            result.append(self._option_head)

            # Print function declaration
            final_function_name = option.decorate(self.function_name)
            fields = []
            # inputs
            in_dim = len(self.input_spec)
            for i in range(in_dim):
                fields.append(VarType1005.const_reference(self.input_spec[i]) + ' ' + self.input_names[i])
            # outputs
            for channel in self.output_channels(option):
                field_type = self.output_channel_type(channel)
                field_name = self.output_channel_name(channel)
                fields.append(field_type + "* " + field_name)
            # constant outputs
            for channel, value in self.constant_derivative_channels.items():
                if value == 0:
                    continue
                field_name = self.output_channel_name(channel)
                field_value = self._constant_output_format % value
                fields.append("/*%s = %s*/" % (field_name, field_value))
            result += self._function_with_fields_to_lines(final_function_name, fields)

            # Empty line
            result.append("")

        return result

    def print_implementation_head(self, option: Option):
        final_function_name = option.decorate(self.function_name)
        fields = []
        # inputs
        in_dim = len(self.input_spec)
        for i in range(in_dim):
            fields.append(VarType1005.const_reference(self.input_spec[i]) + ' ' + self.input_names[i])
        # outputs
        for channel in self.output_channels(option):
            field_type = self.output_channel_type(channel)
            field_name = self.output_channel_name(channel)
            fields.append(field_type + "* " + field_name)
        result = self._function_with_fields_to_lines(final_function_name, fields)

        if result[-1][-1] == ";":
            result[-1] = result[-1][:-1] + " {"
        return result

    @staticmethod
    def _function_with_fields_to_lines(function_final_name: str, fields: List[str]):
        """
        :param function_final_name:
        :param fields: each element shown in separate line, comma free, can be /*???*/.
        MUST avoid extra space in the front.
        :return:
        """
        result = []

        if not fields:
            return ['void ' + function_final_name + "();"]
        elif len(fields) == 1:
            return ['void %s(%s);' % (function_final_name, fields[0])]

        def is_comment_field(field: str):
            return field[:2] == "/*"

        last_non_comment_field = -1
        for i in range(len(fields)):
            if not is_comment_field(fields[i]):
                last_non_comment_field = i

        def may_add_comma(field_id: int):
            field = fields[field_id]
            if is_comment_field(field):
                # no comma for comment field
                return field
            elif field_id == last_non_comment_field:
                return field
            else:
                return field + ","

        # at least two fields
        head = 'void ' + function_final_name + "("
        result.append(head + may_add_comma(0))
        var_spaces = " " * len(head)

        for i in range(1, len(fields) - 1):
            result.append(var_spaces + may_add_comma(i))

        result.append(var_spaces + fields[-1] + ");")

        return result

    @staticmethod
    def _lines_to_function_and_fields(lines: List[str]):
        """
        :param lines: some result generated by  function_with_fields_to_lines
        MUST avoid empty lines.
        :return: function_final_name, fields
        """
        assert lines, "input can't be empty"

        def rms(line):
            return remove_end_spaces(remove_front_spaces(line))

        def rmc(line):
            return line[:-1] if line[-1] == "," else line

        head_with_first_field = rms(lines[0])
        assert head_with_first_field[:5] == "void ", "invalid arg"
        left_bra = head_with_first_field.find("(")
        assert left_bra > 5
        function_final_name = rms(head_with_first_field[5:left_bra])

        fields = []
        should_keep_parse = False
        rest_of_first_line = rms(head_with_first_field[left_bra + 1:])
        if rest_of_first_line[-2:] == ");":
            content = rest_of_first_line[:-2]
            if rms(content) == "":
                pass
            else:
                fields.append(content)
        else:
            # at least two fields
            fields.append(rmc(rest_of_first_line))
            should_keep_parse = True

        if should_keep_parse:
            for i in range(1, len(lines)):
                f_line = rms(lines[i])
                if f_line[-2:] != ");":
                    fields.append(rmc(f_line))
                else:
                    fields.append(rmc(f_line[:-2]))
                    break

        return function_final_name, fields

    def from_header_core(self, lines: List[str]):
        """
        from .h
        A rule: when you can't see
        :return:
        """

        # Step 1 extract supported options and locate the header of each option.
        self.supported_options = []
        option_header_start = []
        option_header_end = []

        def get_header_lines(option_id):
            res = []
            for i in range(option_header_start[option_id],
                           option_header_end[option_id]):
                if lines[i] != "":
                    res.append(lines[i])
            return res

        def rms(line):
            return remove_end_spaces(remove_front_spaces(line))

        is_parsing_option = False
        option_lines = []
        for i in range(len(lines)):
            line = remove_end_spaces(remove_front_spaces(lines[i]))
            is_head = line == self._option_head

            if is_parsing_option:
                if not is_head:
                    option_lines.append(line)
                else:
                    option = Option()
                    option.from_string("\n".join(option_lines), True)
                    self.supported_options.append(option)
                    option_header_start.append(i + 1)
                    is_parsing_option = False
                    option_lines = []
            else:
                if is_head:
                    is_parsing_option = True
                    if option_header_start:
                        option_header_end.append(i)
        option_header_end.append(len(lines))
        assert len(option_header_start) == len(option_header_end)

        # Step 2. parse default option for input and output names specs.
        default_option = Option()
        assert default_option in self.supported_options, "header must have default option"

        default_option_lines = get_header_lines(self.supported_options.index(default_option))
        self.function_name, default_fields = self._lines_to_function_and_fields(default_option_lines)
        self.input_names = []
        self.input_spec = []
        self.output_names = []
        self.output_spec = []

        for field in default_fields:
            # only two cases: input, output
            star = field.find("*")
            if star >= 0:
                # output
                self.output_names.append(rms(field[star + 1:]))
                self.output_spec.append(rms(field[:star]))
            else:
                pure_field = rms(field)
                space = pure_field.find(" ")
                assert space > 0, "invalid field"
                self.input_names.append(rms(pure_field[space + 1:]))
                self.input_spec.append(rms(pure_field[:space]))

        self.safe_check()

        # Step 3. parse all other options to collect constant derivative outputs,
        # How do you know those zero neglected? for each option, their full output channels are different, use that.
        self.constant_derivative_channels = {}
        for i in range(len(self.supported_options)):
            option = self.supported_options[i]
            head = get_header_lines(i)

            _, fields = self._lines_to_function_and_fields(head)

            existing_output_channels = set()
            for field in fields:
                # 3 type of fields: input, output, const derivative output.
                pure_field = rms(field)
                if pure_field[:2] == "/*" and pure_field[-2:] == "*/":
                    # constant output
                    const_output_field = pure_field[2:-2]
                    equator = const_output_field.find("=")
                    assert equator >= 0, "invalid field"
                    field_name = rms(const_output_field[:equator])
                    value = float(const_output_field[equator + 1:])

                    channel = self.output_channel_from_name(field_name)
                    existing_output_channels.add(channel)
                    self.constant_derivative_channels[channel] = value
                else:
                    star = pure_field.find("*")
                    if star >= 0:
                        # normal output
                        field_name = rms(pure_field[star + 1:])
                        existing_output_channels.add(self.output_channel_from_name(field_name))

            missing_channels = self.find_missing_output_channels(option, existing_output_channels)
            for channel in missing_channels:
                self.constant_derivative_channels[channel] = 0.0

    @staticmethod
    def create_from_header_core(lines: List[str]) -> "Header":
        header = Header("", [], [], [], [], [], {}, False)
        header.from_header_core(lines)
        return header

    def print_call(self, full_context: FullContext, prefix="") -> CallResult:
        """
        GongJuRen
        :param prefix: The calling prefix, for e.g.
        SomeNameSpace::Foo(...)
        :param full_context:
        :return:
        """
        result = CallResult()

        # determine the interface
        interface_output_channels = self.output_channels(full_context.option)
        output_channels_required_by_graph = set(full_context.required_output_channels())
        output_field_names = []
        # type to name
        unused_variables: Dict[str, str] = {}
        for channel in interface_output_channels:
            if channel in output_channels_required_by_graph:
                output_field_names.append(full_context.output_channel_name(channel))
            else:
                var_type = self.output_channel_type(channel)
                if var_type not in unused_variables:
                    unused_variables[var_type] = Const1005.graph_unused_prefix + var_type
                output_field_names.append(unused_variables[var_type])

        result.lines = []
        result.lines.append("{")
        for var_type, var_name in unused_variables.items():
            result.lines.append(Const1005.indent + var_type + " " + var_name + ";")

        calling = Const1005.indent + prefix + full_context.option.decorate(self.function_name)

        calling += "("

        # TODO(huaiyuan):split into multiple lines if possible
        # fill input graph vars.
        for i in range(len(full_context.context.input_variables)):
            calling += full_context.context.input_variables[i].nick_name + ","

        for out_name in output_field_names:
            calling += "&" + out_name + ","

        if calling[-1] == ",":
            calling = calling[:-1]
        calling += ");"

        result.lines.append(calling)

        result.lines.append("}")

        # constant_derivative_outputs.
        result.constant_output_channels = {}
        for channel, value in self.constant_derivative_channels.items():
            # here, I guess it is correct
            if channel in output_channels_required_by_graph:
                result.constant_output_channels[channel] = value

        return result

# TODO(make empty header for user)
