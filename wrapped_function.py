from t1005_graph import *
from header import Header
from cpp_library import UserLibrary
import os


class WrappedFunction(FunctionBase):
    """
    Will create wrapped function.
    WILL assume all channels are differentiable scalar/array.
    So user should make sure the function to be wrapped is
    not responsive to those configs.
    """

    def __init__(self, function_to_be_wrapped: FunctionBase,  # const
                 function_name: str,
                 input_names: List[str] = None,
                 output_names: List[str] = None,
                 required_options: Set[Option] = None):
        in_dim = len(function_to_be_wrapped.input_spec)
        out_dim = len(function_to_be_wrapped.output_spec)

        if input_names is None:
            input_names = ["input_%d" % i for i in range(in_dim)]
        if output_names is None:
            output_names = ["output_%d" % i for i in range(out_dim)]
        if required_options is None:
            required_options = AllOptions.all_option_unordered_set.copy()

        # checks
        assert all([is_valid_lower_case_cpp_name(name) for name in input_names])
        assert all([is_valid_lower_case_cpp_name(name) for name in output_names])
        input_spec = function_to_be_wrapped.input_spec.copy()
        output_spec = function_to_be_wrapped.output_spec.copy()
        assert all(
            [spec != '' for spec in input_spec]), "wrapper only accept function with deterministic input spec"
        assert all(
            [spec != '' for spec in output_spec]), "wrapper only accept function with deterministic output spec"
        assert len(input_names) == in_dim
        assert len(output_names) == out_dim

        # Build a graph and call the function
        wrapper_graph = Graph(name="wrapper")
        input_vars = []
        for i in range(in_dim):
            if VarType1005.is_numerical_var_type(input_spec[i]):
                input_vars.append(wrapper_graph.state_inputs([input_names[i], ], input_spec[i]))
            else:
                input_vars.append(wrapper_graph.config_inputs([input_names[i], ], input_spec[i]))
        output_vars = function_to_be_wrapped(*input_vars)

        # Ensure iterable
        if out_dim == 1:
            output_vars = (output_vars,)

        for i in range(out_dim):
            output_vars[i].set_name(Const1005.wrapper_graph_output_prefix + output_names[i])
        context = Context(input_vars, list(output_vars))

        # Extract print_call results
        constant_derivative_channels = {}
        self.options = list(required_options)
        self.options.sort(key=lambda option: option.to_string())
        self.call_results = []
        for option in self.options:
            full_context = FullContext(context, option)
            result = function_to_be_wrapped.print_call(full_context)

            for channel in full_context.non_required_output_channels():
                constant_derivative_channels[channel] = 0.0
            for channel, value in result.constant_output_channels.items():
                constant_derivative_channels[channel] = value

            self.call_results.append(result)
        self.context = context
        # Extract constant derivative outputs.

        self.header = Header(function_name, self.options.copy(), input_spec.copy(), output_spec.copy(),
                             input_names.copy(), output_names.copy(), constant_derivative_channels)

        # prepare definition
        self._definition_results_per_option = [DefinitionResult()] * len(self.options)
        for i in range(len(self.options)):
            option = self.options[i]
            result = DefinitionResult()
            final_name = option.decorate(self.header.function_name)
            result.function_names_to_lines[final_name] = self._print_implementation(i)
            self._definition_results_per_option.append(result)

        super(WrappedFunction, self).__init__(input_spec, output_spec,
                                              function_to_be_wrapped.dependencies,
                                              required_options)

    def _print_implementation(self, option_id) -> List[str]:
        option = self.options[option_id]
        result = []

        result += self.header.print_implementation_head(option)
        call_result = self.call_results[option_id]
        full_context = FullContext(self.context, option)

        # Name of input in header is in accordance with full_context
        # Name of output in header is not.
        result.append(Const1005.indent + "// Link interface outputs to wrapper graph.")
        for channel in self.header.output_channels(option):
            head_name = self.header.output_channel_name(channel)
            head_type = self.header.output_channel_type(channel)

            context_name = full_context.output_channel_name(channel)
            result.append(Const1005.indent + head_type + "& " + context_name + "=*" + head_name + ";")

        for ln in call_result.lines:
            result.append(Const1005.indent + ln)

        result.append("}")

        return result

    def get_definition(self, option: Option) -> DefinitionResult:
        assert option in self.supported_options
        option_id = self.options.index(option)
        return self._definition_results_per_option[option_id]

    def print_call(self, full_context: FullContext) -> CallResult:
        return self.header.print_call(full_context)

    def optional_header(self) -> "Header":
        return self.header

    # TODO(huaiyuan): support dumping multiple functions
    def dump_to_lib(self, library: UserLibrary, force_update=True,
                    namespace: List[str] = None):
        path = library.lib_abs_path()
        name = library.lib_name()
        header_file = os.path.join(path, name + ".h")
        cpp_file = os.path.join(path, name + ".cpp")

        if not os.path.exists(path) and force_update:
            os.makedirs(path)
        if os.path.exists(header_file) and force_update:
            os.remove(header_file)
        if os.path.exists(cpp_file) and force_update:
            os.remove(cpp_file)

        assert os.path.exists(path), "<%s> not exist." % path
        assert not os.path.exists(header_file), "<%s> already exist." % header_file
        assert not os.path.exists(cpp_file), "<%s> already exist." % cpp_file

        if namespace is None:
            namespace = []

        def write_lines(lines: List[str], indent=0):
            for l in lines:
                fp.write(str(Const1005.indent * indent) + l + "\n")

        def empty_line():
            fp.write("\n")

        namespace_string = "::".join(namespace)

        with open(header_file, "w") as fp:
            write_lines(library.head_comments_h())
            empty_line()

            # includes
            includes = []
            for dep in self.dependencies:
                includes.append("#include" + dep.include_name())
            write_lines(includes)
            empty_line()

            if namespace_string != "":
                write_lines(["namespace %s {" % namespace_string])
            # header core
            write_lines([library.header_begin_comments_h()])
            write_lines(self.header.print_header_core(), indent=1 if namespace_string != "" else 0)
            write_lines([library.header_end_comments_h()])

            if namespace_string != "":
                write_lines(["}  // namespace %s" % namespace_string])
            write_lines(library.tail_comments_h())
            empty_line()

        with open(cpp_file, "w") as fp:

            # includes
            includes = ["#include" + library.include_name()]
            write_lines(includes)
            empty_line()

            if namespace_string != "":
                write_lines(["namespace %s {" % namespace_string])
            # header core

            for i in range(len(self.options)):
                write_lines(self._print_implementation(i), indent=1 if namespace_string != "" else 0)
                empty_line()

            if namespace_string != "":
                write_lines(["}  // namespace %s" % namespace_string])
            write_lines(library.tail_comments_h())
            empty_line()


def wrap_graph_function(graph_function: GraphFunction, function_name: str) -> WrappedFunction:
    return WrappedFunction(graph_function,
                           function_name,
                           [variable.nick_name for variable in graph_function.graph_input_variables],
                           # It is in principle, safe to not add the out_ prefix here.
                           # There are 2 scopes for wrapper: wp graph and interface.
                           ["out_" + variable.nick_name for variable in graph_function.graph_output_variables],
                           required_options=graph_function.supported_options)


def wrap_graph(graph: Graph,
               input_variables: List[Variable],
               output_variables: List[Variable],
               function_name: str) -> WrappedFunction:
    return wrap_graph_function(graph.create_graph_function(input_variables, output_variables), function_name)
