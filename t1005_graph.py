from typing import List, Tuple, Dict, Set, Any
import sympy as sp
from cpp_library import UserLibrary, CppLibrary
from sympy.printing.cxx import CXX11CodePrinter
from common import *

from t1005_option import Option, AllOptions


class Variable:
    # rear can convert into front, vise NOT versa
    # TODO(): Consider whether it is possible that one claim ArrayXd var = 1.0;
    # numerical_var_types = ['ArrayXd', 'double', 'float', 'int']
    # numerical_var_types_set = set(numerical_var_types)

    # A more complex type can be initialized from a less complex one.
    # types with equal complexity are inter-tranferable.
    numerical_var_type_complexity = {'int': 1,
                                     'float': 2,
                                     'double': 3,
                                     'ArrayXd': 4}

    var_types_not_need_const_reference = ['double', 'int']
    default_var_type = 'double'

    TYPE_UN_DEFINED = -1
    TYPE_STATE_INPUT = 0
    TYPE_CONSTANT = 1
    TYPE_CONFIG_INPUT = 2
    TYPE_STATE_EXPR = 3
    TYPE_CONFIG_EXPR = 4
    TYPE_CONSTANT_EXPR = 5

    def __init__(self, name: str, nick_name: str, graph: "Graph"):
        self.name = name
        self.nick_name = nick_name
        self.graph = graph

        self.var_type = self.default_var_type
        self.type = self.TYPE_UN_DEFINED

    def is_differentiable(self):
        return self.type in [self.TYPE_STATE_EXPR, self.TYPE_STATE_INPUT]

    def set_name(self, name: str):
        self.graph.re_name(self.nick_name, name)

    def defined_as_state_input(self, var_type: str = 'double'):
        assert self.type is self.TYPE_UN_DEFINED, "re definition not allowed"
        assert VarType1005.is_numerical_var_type(var_type), "Invalid type: %s" % var_type

        self.type = self.TYPE_STATE_INPUT
        self.var_type = var_type

    def defined_as_constant(self, value: float, var_type: str = 'double'):
        """
        :param value:
        :param var_type:
        :return:
        """
        assert self.type is self.TYPE_UN_DEFINED, "re definition not allowed"
        assert VarType1005.is_numerical_var_type(var_type), "Invalid type: %s" % var_type

        self.value = value
        self.type = self.TYPE_CONSTANT
        self.var_type = var_type

    def defined_as_config_input(self, var_type: str = 'double'):
        """
        :param var_type:
        :return:
        """
        assert self.type is self.TYPE_UN_DEFINED, "re definition not allowed"
        assert is_valid_cpp_name(var_type), "Invalid type: %s" % var_type

        self.type = self.TYPE_CONFIG_INPUT
        self.var_type = var_type

    def defined_as_expr(self, function: "FunctionBase", inputs: List["Variable"], output_channel=0,
                        var_type: str = 'double'):
        """
        :param function:
        :param inputs:
        :param output_channel:
        :param var_type:
        :return:
        """
        assert self.type is self.TYPE_UN_DEFINED, "re definition not allowed"
        assert VarType1005.is_numerical_var_type(var_type), "Invalid type: %s" % var_type

        input_types = [input_variable.type for input_variable in inputs]
        assert Variable.TYPE_UN_DEFINED not in input_types
        if Variable.TYPE_STATE_EXPR in input_types or Variable.TYPE_STATE_INPUT in input_types:
            output_type = Variable.TYPE_STATE_EXPR
        elif Variable.TYPE_CONFIG_EXPR in input_types or Variable.TYPE_CONFIG_INPUT in input_types:
            output_type = Variable.TYPE_CONFIG_EXPR
        else:
            output_type = Variable.TYPE_CONSTANT_EXPR

        self.function = function
        self.inputs = inputs.copy()
        self.output_channel = output_channel
        self.type = output_type
        self.var_type = var_type

    # Some basic operators
    # not recommended for massive use
    def __add__(self, other):
        assert False, "not implemented"
    def __radd__(self, other):
        assert False, "not implemented"

    def __sub__(self, other):
        assert False, "not implemented"

    def __rsub__(self, other):
        assert False, "not implemented"

    def __mul__(self, other):
        assert False, "not implemented"
    def __rmul__(self, other):
        assert False, "not implemented"

    def __pow__(self, other):
        assert False, "not implemented"

    def __rpow__(self, other):
        assert False, "not implemented"

    def __truediv__(self, other):
        assert False, "not implemented"

    def __rtruediv__(self, other):
        assert False, "not implemented"


class Context:
    """
    Function context specifies the major input output action.

    For how to call the function, it depends on the class.
    """

    def __init__(self, inputs: List[Variable], outputs: List[Variable]):
        self.input_variables = inputs
        self.output_variables = outputs


class FullContext:
    def __init__(self, context: Context,
                 option: Option):
        self.context = context
        self.option = option
        # and other options maybe

        self.zero_order_channels: List[Tuple[int]] = []
        self.first_order_channels: List[Tuple[int, int]] = []
        self.second_order_channels: List[Tuple[int, int, int]] = []

        self.non_required_channels: List[Tuple] = []
        self._init_output_channels()

    def _init_output_channels(self):
        """
        keep only the differentiable output channels
         0 order non-differentiable also kept.
        :return: List[Tuple]
        """
        out_dim = len(self.context.output_variables)
        in_dim = len(self.context.input_variables)

        full_output_channels = full_output_channels_with_derivatives(in_dim, out_dim,
                                                                     self.option.enable_1st_order_derivative(),
                                                                     self.option.enable_2nd_order_derivative())

        for channel in full_output_channels:
            # keep only the differentiable output channels
            # 0 order non-differentiable also kept.
            should_avoid_channel = False
            if len(channel) > 1:
                if not self.context.output_variables[channel[0]].is_differentiable():
                    should_avoid_channel = True

                for i in range(1, len(channel)):
                    if not self.context.input_variables[channel[i]].is_differentiable():
                        should_avoid_channel = True
            else:
                # 0 order channel always kept.
                pass

            if should_avoid_channel:
                self.non_required_channels.append(channel)
            else:
                if len(channel) == 1:
                    self.zero_order_channels.append(channel)
                elif len(channel) == 2:
                    self.first_order_channels.append(channel)
                elif len(channel) == 3:
                    self.second_order_channels.append(channel)

    def required_output_channels(self):
        return self.zero_order_channels + \
               self.first_order_channels + \
               self.second_order_channels

    def non_required_output_channels(self):
        """
        The complimentary of required output channels.
        """
        return self.non_required_channels

    def output_channel_name(self, channel: Tuple):
        assert len(channel) in {1, 2, 3}
        if len(channel) > 1:
            ch_names = [self.context.output_variables[channel[0]].nick_name] + \
                       ["%s%d" % (Const1005.input_channel_short, i) for i in channel[1:]]
            return Const1005.node_derivative_prefix + get_channel_name(tuple(ch_names))
        else:
            return self.context.output_variables[channel[0]].nick_name

    def output_channel_type(self, channel: Tuple):
        assert len(channel) > 0
        return self.context.output_variables[channel[0]].var_type


class CallResult:
    def __init__(self):
        self.lines: List[str] = []
        self.constant_output_channels: Dict[Tuple, float] = {}


class DefinitionResult:
    def __init__(self):
        self.function_names_to_lines: Dict[str, List[str]] = {}


class FunctionBase:
    """
    TODO(huaiyuan): does it support an option? answer in virtual.
    """

    def __init__(self, input_spec: List[str], output_spec: List[str], dependencies: Set[CppLibrary] = None,
                 supported_options: Set[Option] = None):
        """
        Use '' to make function infer the type during call.
        :param output_spec: ['double','double']
        :param input_spec: ['double','float','']
        """

        assert all([(is_valid_cpp_name(var_type) or var_type == '') for var_type in input_spec]), \
            "invalid input spec"
        assert all([(VarType1005.is_numerical_var_type(var_type) or var_type == '') for var_type in output_spec]), \
            "invalid output spec"

        self.output_spec = output_spec.copy()
        self.input_spec = input_spec.copy()
        self.dependencies = set() if dependencies is None else dependencies.copy()
        self.supported_options = AllOptions.all_option_unordered_set if supported_options is None else supported_options.copy()

    def __call__(self, *args, **kwargs):
        # Check graph consistency
        graph: "Graph" = None
        for element in args:
            if type(element) is Variable:
                if graph is None:
                    graph = element.graph
                else:
                    assert graph is element.graph, "Function can't be called on variables from different graphs."
        assert graph is not None, "Can't infer graph from function call."

        existing_input_numerical_types = []
        for element in args:
            if type(element) is Variable:
                existing_input_numerical_types.append(element.var_type)
        inferred_var_type = VarType1005.infer_combined_var_type(existing_input_numerical_types)

        # Create input variables
        input_variables: List[Variable] = []
        for element in args:
            if type(element) is Variable:
                input_variables.append(element)
            elif type(element) in [float, int]:
                index = len(input_variables)
                var_type = inferred_var_type if self.input_spec[index] == '' else self.input_spec[index]
                assert var_type is not None, "Function with no numerical input can't have unspecified input type."

                unnamed_variable = graph.create_un_named_variable()
                unnamed_variable.defined_as_constant(element, var_type)
                input_variables.append(unnamed_variable)

        # Create output variables
        output_variables: List[Variable] = []
        for i in range(len(self.output_spec)):
            var_type = inferred_var_type if self.output_spec[i] == '' else self.output_spec[i]
            assert var_type is not None, "Function with no numerical input can't have unspecified output type."
            output_variable = graph.create_un_named_variable()
            output_variable.defined_as_expr(self, input_variables, i, var_type)
            output_variables.append(output_variable)

        # check the context compatibility
        context = Context(input_variables, output_variables)
        res, dbg = self.is_compatible(context)
        assert res, dbg

        # Inform graph the operation.
        graph.append_operation(self, context)

        if len(output_variables) > 1:
            return tuple(output_variables)
        else:
            return output_variables[0]

    def is_compatible(self, context: Context) -> Tuple[bool, str]:
        """
        :param context:
        :return: result, debug_string
        """
        out_dim = len(context.output_variables)
        in_dim = len(context.input_variables)

        if len(self.output_spec) != out_dim:
            return False, "output variable number miss-match"
        if len(self.input_spec) != in_dim:
            return False, "input variable number miss-match"

        # We check the type matching
        for i in range(out_dim):
            spec = self.output_spec[i]
            output_var_type = context.output_variables[i].var_type
            if spec == '':
                continue
            if spec == output_var_type or VarType1005.is_transferable(spec, output_var_type):
                continue
            return False, "The %d' th output type mis-match: require %s, got %s" % (i, spec, output_var_type)

        for i in range(in_dim):
            spec = self.input_spec[i]
            input_var_type = context.input_variables[i].var_type
            if spec == '':
                continue
            if spec == input_var_type or VarType1005.is_transferable(input_var_type, spec):
                continue
            return False, "The %d' th input type mis-match: require %s, got %s" % (i, spec, input_var_type)

        # check output TYPE to be EXPR
        for output_var in context.output_variables:
            if output_var.type not in [Variable.TYPE_STATE_EXPR, Variable.TYPE_CONFIG_EXPR]:
                return False, "output variables must be expr"
        return True, ""

    def get_definition(self, option: Option) -> DefinitionResult:
        """
        What appears before the graph.
        For e.g. for sympy nodes with large result, we want to wrap them using WrappedFunction

        Will enable this when:
        :param inputs: a, b, c
        :param outputs: sum, mult
        :return: List[str] as lines.

        inline ArrayNd GetSumMultFromABC(double a, const ArrayNd& b, double c) {
            ...
            ...
        }

        """
        return DefinitionResult()

    def print_call(self, full_context: FullContext) -> CallResult:
        """
        Will:
        Under normal cpp 1-2 order derivative options:
        1, assume the inputs variables are defined and is ready.
        2, assume the non-constant output derivatives are defined yet not assigned with value.
        3, make sure each field in full_context.required_output_channels() are answered by:
            a, assume the required output chl variable is defined and simply assign value to it in lines.
            b, tell the user that this output chl is constant using constant_outputs.

        The function will print lines in C++ to calculate the output variables and non-constant derivatives.
        returned as List[str]
        The function will tell user which derivatives are constant and should not be defined.
        returned as Dist[str, float]
        Notice that user should call this function FIRST to know what to define.
        """
        assert False, "not implemented"
        return CallResult()

    def optional_header(self) -> "Header":
        """
        return header if function has a header
        """
        return None


# the class that is ready to be

# Most functions should NOT provide constant outputs.
# this is because the interface/implementation would be dependent upon the context.

# so what we need is: context independent constant outputs.

# we need both context and function detail to determine whether an output der is constant.


# A convention:
# When Function deems whether an input is derivative-interested by itself.
# For a derivative that is not printed, it would be 0.0.
# A function also need to make sure what it prints is not pre-defined.
# this is achieved by assuming the naming unique-ness of input variables.

# What print call do: provide the

"""
class CppMemberFunction(FunctionBase):
    def __init__(self, object_type: str, object_name: str, function_name: str, input_spec: List[str],
                 output_spec: List[str]):

        const CostSpace& space1;
        space1.collision_cost(...)
        :param object_type: CostSpace
        :param object_name: space1
        :param function_name: collision_cost
        # input , output types must be specified
        assert all([var_type in Variable.numerical_var_types for var_type in input_spec]), "invalid input spec"
        assert all([var_type in Variable.numerical_var_types for var_type in output_spec]), "invalid output spec"

        super(CppMemberFunction, self).__init__(input_spec, output_spec)

        assert _is_valid_namespace(object_type), "invalid object_type :(%s)" % object_type
        assert _is_valid_cpp_name(object_name), "invalid object_name :(%s)" % object_name
        assert _is_valid_cpp_name(function_name), "invalid function_name :(%s)" % function_name

        self.object_type = object_type
        self.object_name = object_name
        self.function_name = function_name
"""


# CppNamespaceFunction: header to include.
# What to have in the cpp file?
# the definition of each channel. I mean, at least you need constant outputs. implementation(assumed)
# there is a comment decoration protocol.
# 1, function name.
# 2, function constant outputs.
# 3, const references, just all what a Graph Function should have.
# 4, input spec.
# 5, output spec.

# interface function !! base class.


class ConditionalFunction(FunctionBase):
    # So we can avoid running a piece of code, by using gflag int.
    # We only support gflag int.
    # TODO(): implement with :
    #  1, checking no - loop dependency
    #  2, it is a little tricky, since each sub function are defining the output vars by themselves,
    #  There can be disagreement in the output var_types and even existence.
    #  That we need to merge them, their result types and names.
    pass


class GraphFieldManager:
    def __init__(self):
        self.existing_fields: Set[str] = set()
        self.constant_fields: Dict[str] = {}
        pass

    def claim_field_as_normal(self, name: str):
        self.existing_fields.add(name)

    def claim_field_as_constant(self, name: str, value: float):
        assert name not in self.constant_fields, "\'%s\' already claimed!" % name
        self.constant_fields[name] = value

    def is_constant(self, name: str):
        if name in self.existing_fields:
            return False
        else:
            return True

    def is_zero(self, name: str):
        if name in self.existing_fields:
            return False
        elif name in self.constant_fields:
            return self.constant_fields[name] == 0
        else:
            return True

    def get_constant_value(self, name: str):
        if name in self.constant_fields:
            return self.constant_fields[name]
        else:
            return 0

    def add_product_of_fields_to_target_field(self,
                                              output_lines: List[str],  # output
                                              target_field: str,
                                              target_field_type: str,
                                              fields_to_prod: List[str],
                                              gain: float = 1,
                                              indent_num: int = 0):
        if not fields_to_prod:
            return
        # at least adding something.

        constant_factor = gain
        for name in fields_to_prod:
            if self.is_constant(name):
                constant_factor *= self.get_constant_value(name)

        items = ''
        for name in fields_to_prod:
            if not self.is_constant(name):
                items += name + '*'
        if items != '':
            items = items[:-1]

        # items and constant factor append to field
        if constant_factor == 0:
            return

        all_indents = Const1005.indent * indent_num

        if items == '':
            # adding constant number
            if target_field in self.constant_fields:
                # add only to storage
                self.constant_fields[target_field] += constant_factor
                # output_lines.append('all_indents + // %s += %f;'%(target_field, constant_factor))
            elif target_field in self.existing_fields:
                # add to variable
                output_lines.append(all_indents + '%s += %f;' % (target_field, constant_factor))
            else:
                # create variable
                self.constant_fields[target_field] = constant_factor
                # output_lines.append('all_indents + // %s = %f;'%(target_field, constant_factor))
        else:
            # adding expression
            expr = "(%f) * %s" % (constant_factor, items) if constant_factor != 1 else items
            if target_field in self.constant_fields:
                # no longer constant, become existing normal
                value = self.constant_fields[target_field]
                full_expr = '%f + %s' % (value, items) if value != 0 else items
                output_lines.append(all_indents + '%s %s=%s;' % (target_field_type, target_field, full_expr))

                del self.constant_fields[target_field]
                self.existing_fields.add(target_field)
            elif target_field in self.existing_fields:
                # add expr to existing field.
                output_lines.append(all_indents + '%s += %s;' % (target_field, expr))
            else:
                # create existing field with expr
                output_lines.append(all_indents + '%s %s=%s;' % (target_field_type, target_field, expr))
                self.existing_fields.add(target_field)


# DONE


class Graph:
    """
    """

    def __init__(self, name: str = "UntitledFunc"):
        # contain all variables
        # from nick_name to Variable
        self.name = name
        self._all_variables: Dict[str, Variable] = {}

        self._un_named_count = 0

        self._operations: List[Tuple[FunctionBase, Context]] = []
        self._all_functions_with_headers: Dict[str, FunctionBase] = {}

    def state_inputs(self, names: List[str], var_type: str):
        assert VarType1005.is_numerical_var_type(var_type), "Invalid type: %s" % var_type
        if not names:
            return None

        results = [self.create_new_variable(name) for name in names]
        for var in results:
            var.defined_as_state_input(var_type)

        if len(results) == 1:
            return results[0]
        else:
            return results

    def config_inputs(self, names: List[str], var_type: str):
        assert is_valid_cpp_name(var_type), "Invalid type: %s" % var_type
        if not names:
            return None

        results = [self.create_new_variable(name) for name in names]
        for var in results:
            var.defined_as_config_input(var_type)

        if len(results) == 1:
            return results[0]
        else:
            return results

    def append_operation(self, function: FunctionBase, context: Context):
        # one graph won't allow two functions with same name.
        header = function.optional_header()
        if header is not None:
            if header.function_name in self._all_functions_with_headers:
                existing_one = self._all_functions_with_headers[header.function_name]
                assert existing_one is function, \
                    "Function object" + str(function) + " and " + str(existing_one) \
                    + "(existing in graph) has the same name:" + header.function_name
            else:
                self._all_functions_with_headers[header.function_name] = function

        res, dbg = function.is_compatible(context)
        assert res, dbg
        # TODO(): check function references are good with variable names.
        self._operations.append((function, context))

    def evaluate_all_dependencies(self) -> Set[CppLibrary]:
        all_deps: Set[CppLibrary] = set()
        for func, _ in self._operations:
            all_deps.update(func.dependencies)
        return all_deps

    def evaluate_all_supported_options(self) -> Set[Option]:
        all_ops = AllOptions.all_option_unordered_set
        for func, _ in self._operations:
            all_ops.intersection_update(func.supported_options)
        return all_ops

    #
    def get_state_input_variables(self):
        res = []
        for variable in self._all_variables.values():
            if variable.type is Variable.TYPE_STATE_INPUT:
                res.append(variable)
        return res

    def get_config_input_variables(self):
        res = []
        for variable in self._all_variables.values():
            if variable.type is Variable.TYPE_CONFIG_INPUT:
                res.append(variable)
        return res

    def is_member(self, variable: Variable):
        if not variable.graph is self:
            return False
        if not variable.nick_name in self._all_variables:
            return False
        if not self._all_variables[variable.nick_name] is variable:
            return False
        return True

    def create_new_variable(self, name: str):
        new_var = self.create_un_named_variable()
        res, dbg = self.re_name(new_var.nick_name, name)
        assert res, dbg

        return new_var

    def re_name(self, old_nick_name: str, new_name: str):
        if old_nick_name not in self._all_variables.keys():
            return False, "\"%s\" not found" % old_nick_name
        new_nick_name = self._get_nick_name(new_name)
        if new_nick_name is None:
            return False, "Invalid name: %s" % new_name
        if new_nick_name in self._all_variables.keys():
            return False, "Nick name \"%s\"->\"%s\" already exist for \"%s\"" % (
                new_name, new_nick_name, self._all_variables[new_nick_name].name)
        for sub_string in Const1005.built_in_name_sub_strings:
            if new_nick_name.find(sub_string) >= 0:
                return False, "Nick name \"%s\"->\"%s\" contains built-in sub string: \"%s\"" % (
                    new_name, new_nick_name, sub_string)

        var = self._all_variables[old_nick_name]

        var.name = new_name
        var.nick_name = new_nick_name
        self._all_variables[new_nick_name] = var
        self._all_variables.pop(old_nick_name)
        return True, ""

    def create_un_named_variable(self):
        nick_name = Const1005.unnamed_graph_var_prefix + "%d" % self._un_named_count
        self._un_named_count += 1

        for sub_string in Const1005.built_in_name_sub_strings:
            assert nick_name.find(
                sub_string) < 0 or sub_string == Const1005.unnamed_graph_var_prefix, "Nick name \"%s\" contains built-in sub string: \"%s\"" % (
                nick_name, sub_string)

        new_variable = Variable(nick_name, nick_name, self)
        self._all_variables[nick_name] = new_variable

        return new_variable

    @staticmethod
    def _get_nick_name(name: str):
        """
        generate a string that is suitable for variable naming as a coding variable
        :param name:
        :return:
        """
        name = name.replace(' ', '_')
        name = name.lower()

        result = ''
        count = 0
        for c in name:
            if c == '_' or c in Const1005.lower_alphabet_chars or (count > 0 and c in Const1005.number_chars):
                result += c
            else:
                result += 'x'
            count += 1

        return result if len(result) > 0 else None

    def get_definition(self,
                       option: Option) -> DefinitionResult:
        all_definitions = DefinitionResult()

        for function, context in self._operations:
            result = function.get_definition(option)
            for name, lines in result.function_names_to_lines.items():
                if name not in all_definitions.function_names_to_lines:
                    all_definitions.function_names_to_lines[name] = lines

        return all_definitions

    # TODO(huaiyuan): comment on implementation: how many addition, how many multiplication
    def print_call(self,
                   outer_full_context: FullContext,
                   self_input_variables: List[Variable],
                   self_output_variables: List[Variable]) -> CallResult:
        option = outer_full_context.option
        supported_options = self.evaluate_all_supported_options()
        assert option in supported_options, "option is not supported <%s>" % option.to_string()

        in_dim = len(self_input_variables)
        out_dim = len(self_output_variables)
        assert in_dim == len(outer_full_context.context.input_variables)
        assert out_dim == len(outer_full_context.context.output_variables)

        for variable in self_output_variables:
            assert self.is_member(variable), "self output variables must be member of the graph."

        for variable in self_input_variables:
            assert self.is_member(variable), "self input variables must be member of the graph."

        names_of_inputs = set()
        for variable in self._all_variables.values():
            if variable.type in {Variable.TYPE_STATE_INPUT, Variable.TYPE_CONFIG_INPUT}:
                names_of_inputs.add(variable.nick_name)
        assert len(self_input_variables) == len(names_of_inputs), \
            "please make sure input_variables exactly contains: " + str(names_of_inputs)
        assert names_of_inputs == {variable.nick_name for variable in self_input_variables}, \
            "please make sure input_variables exactly contains: " + str(names_of_inputs)

        result = CallResult()

        def append_line(line, indent_num=0):
            result.lines.append(Const1005.indent * indent_num + line)

        # this is a hack
        def insert_lines_to_bracket_begin(lines: List[str]):
            result.lines = result.lines[:1] + lines + result.lines[1:]

        append_line("{")

        # input names of outer graph and inner graph can overlap.
        # link inputs to make sure graph input vars are ready.
        inputs_need_link = []
        for i in range(in_dim):
            rhs = outer_full_context.context.input_variables[i].nick_name
            lhs = self_input_variables[i].nick_name
            if rhs != lhs:
                inputs_need_link.append(i)

        append_line("// Link inputs to conveyor variables", indent_num=1)
        for i in inputs_need_link:
            rhs = outer_full_context.context.input_variables[i].nick_name
            mid = Const1005.graph_input_conveyor_prefix + "%d" % i
            var_type = self_input_variables[i].var_type
            append_line(VarType1005.const_reference(var_type) + " " + mid + " = " + rhs + ";", indent_num=1)

        append_line("// Link conveyors to graph variables", indent_num=1)
        for i in inputs_need_link:
            mid = Const1005.graph_input_conveyor_prefix + "%d" % i
            lhs = self_input_variables[i].nick_name
            var_type = self_input_variables[i].var_type
            append_line(VarType1005.const_reference(var_type) + " " + lhs + " = " + mid + ";", indent_num=1)
        append_line("", indent_num=1)
        append_line("// Graph operations", indent_num=1)

        # when asked 2nd order derivative, 1st order must be also ready.
        sub_function_option = Option(
            enable_1st_order_derivative=option.enable_2nd_order_derivative() or option.enable_1st_order_derivative(),
            enable_2nd_order_derivative=option.enable_2nd_order_derivative())
        assert sub_function_option in AllOptions._all_options, "sub option Must be one of _all_options"

        manager = GraphFieldManager()

        self_output_variables_unreached = {self_output_variables[i].nick_name for i in range(out_dim)}
        for function, context in self._operations:
            full_context = FullContext(context,
                                       sub_function_option)

            call_result = \
                function.print_call(full_context)
            # TODO(): clear unused variables making use of AC automaton.

            for channel in full_context.required_output_channels():
                result_name = full_context.output_channel_name(channel)
                result_type = full_context.output_channel_type(channel)
                if channel not in call_result.constant_output_channels:
                    append_line(result_type + ' ' + result_name + ';', indent_num=1)
                    manager.claim_field_as_normal(result_name)
                else:
                    manager.claim_field_as_constant(result_name, call_result.constant_output_channels[channel])

            for ln in call_result.lines:
                append_line(ln, indent_num=1)

            for output_variable in full_context.context.output_variables:
                self_output_variables_unreached.discard(output_variable.nick_name)
            if not self_output_variables_unreached:
                # early stop since all variables collected
                break

        def get_graph_derivative_name(out_name: str, in_names: List[str]):
            in_names_copy = in_names.copy()
            in_names_copy.sort()

            order = len(in_names_copy)
            assert order in [1, 2]

            if order is 1:
                return Const1005.graph_derivative_prefix + get_channel_name((out_name, in_names_copy[0]))
            elif order is 2:
                return Const1005.graph_derivative_prefix + get_channel_name((out_name, *in_names_copy))
            else:
                assert False

        if option.enable_1st_order_derivative() or option.enable_2nd_order_derivative():
            # Compute first order derivatives
            for active_variable in self_output_variables:
                if not active_variable.is_differentiable():
                    continue
                # determine derivative types
                active_var_type = active_variable.var_type

                # skip steps after getting the active variable
                step = len(self._operations) - 1
                while step >= 0 and active_variable not in self._operations[step][1].output_variables:
                    step -= 1

                # starts from the fact da_da = 1.
                manager.claim_field_as_constant(
                    get_graph_derivative_name(active_variable.nick_name, [active_variable.nick_name, ]), 1)

                # bp the entire graph
                for i in range(step, -1, -1):
                    _, context = self._operations[i]
                    full_context = FullContext(context,
                                               sub_function_option)

                    # Update graph derivative from Node derivatives
                    for out_idx, in_idx in full_context.first_order_channels:
                        node_der_name = full_context.output_channel_name((out_idx, in_idx))
                        in_name = context.input_variables[in_idx].nick_name
                        out_name = context.output_variables[out_idx].nick_name

                        # Simple chain rule
                        d_active_d_node_in = get_graph_derivative_name(active_variable.nick_name, [in_name, ])
                        d_active_d_node_out = get_graph_derivative_name(active_variable.nick_name, [out_name, ])

                        manager.add_product_of_fields_to_target_field(result.lines,
                                                                      d_active_d_node_in, active_var_type,
                                                                      [d_active_d_node_out, node_der_name],
                                                                      indent_num=1)

        if option.enable_2nd_order_derivative():
            # A table of co-relation.
            # ensures x not in table[x]
            def co_relate(name_1, name_2, table: Dict[str, Set[str]]):
                if name_1 == name_2:
                    return
                if name_1 not in table:
                    table[name_1] = set()
                if name_2 not in table:
                    table[name_2] = set()

                table[name_1].add(name_2)
                table[name_2].add(name_1)

            for active_variable in self_output_variables:
                if not active_variable.is_differentiable():
                    continue

                # The cross items of a certain variable
                existing_cross_items_of_variable: Dict[str, Set[str]] = {}

                # determine derivative types
                active_var_type = active_variable.var_type

                # skip steps after getting the active variable
                step = len(self._operations) - 1
                while step >= 0 and active_variable not in self._operations[step][1].output_variables:
                    step -= 1

                # bp the entire graph
                for i in range(step, -1, -1):
                    _, context = self._operations[i]
                    full_context = FullContext(context,
                                               sub_function_option)

                    # Update graph derivative from Node derivatives
                    for out_idx, in_idx_1, in_idx_2 in full_context.second_order_channels:
                        in_1_name = full_context.context.input_variables[in_idx_1].nick_name
                        in_2_name = full_context.context.input_variables[in_idx_2].nick_name
                        out_name = full_context.context.output_variables[out_idx].nick_name

                        d2_active_d_node_in_1_d_node_in_2 = \
                            get_graph_derivative_name(active_variable.nick_name, [in_1_name, in_2_name])
                        d_active_d_node_out = get_graph_derivative_name(active_variable.nick_name, [out_name, ])
                        d2_active_d_node_out_d_node_out = \
                            get_graph_derivative_name(active_variable.nick_name, [out_name, out_name])

                        node_d2_out_d_in_1_d_in_2 = full_context.output_channel_name((out_idx, in_idx_1, in_idx_2))
                        node_d_out_d_in_1 = full_context.output_channel_name((out_idx, in_idx_1))
                        node_d_out_d_in_2 = full_context.output_channel_name((out_idx, in_idx_2))

                        manager.add_product_of_fields_to_target_field(result.lines,
                                                                      d2_active_d_node_in_1_d_node_in_2,
                                                                      active_var_type,
                                                                      [d_active_d_node_out, node_d2_out_d_in_1_d_in_2],
                                                                      indent_num=1)

                        manager.add_product_of_fields_to_target_field(result.lines,
                                                                      d2_active_d_node_in_1_d_node_in_2,
                                                                      active_var_type,
                                                                      [d2_active_d_node_out_d_node_out,
                                                                       node_d_out_d_in_1, node_d_out_d_in_2],
                                                                      indent_num=1)

                        if in_idx_1 != in_idx_2 and \
                                not manager.is_zero(d2_active_d_node_in_1_d_node_in_2):
                            co_relate(in_1_name, in_2_name, existing_cross_items_of_variable)

                    for out_idx, in_idx in full_context.first_order_channels:
                        in_name = full_context.context.input_variables[in_idx].nick_name
                        out_name = full_context.context.output_variables[out_idx].nick_name

                        # output channel has no co-related items
                        if out_name not in existing_cross_items_of_variable:
                            continue

                        node_d_out_d_in = full_context.output_channel_name((out_idx, in_idx))
                        for co_related in existing_cross_items_of_variable[out_name]:
                            d2_active_d_node_out_d_co_related = \
                                get_graph_derivative_name(active_variable.nick_name, [out_name, co_related])
                            d2_active_d_node_in_d_co_related = \
                                get_graph_derivative_name(active_variable.nick_name, [in_name, co_related])
                            if co_related != in_name:
                                manager.add_product_of_fields_to_target_field(result.lines,
                                                                              d2_active_d_node_in_d_co_related,
                                                                              active_var_type,
                                                                              [d2_active_d_node_out_d_co_related,
                                                                               node_d_out_d_in],
                                                                              indent_num=1)
                            else:
                                manager.add_product_of_fields_to_target_field(result.lines,
                                                                              d2_active_d_node_in_d_co_related,
                                                                              active_var_type,
                                                                              [d2_active_d_node_out_d_co_related,
                                                                               node_d_out_d_in],
                                                                              indent_num=1,
                                                                              gain=2)

                            if co_related != in_name and \
                                    not manager.is_zero(d2_active_d_node_in_d_co_related):
                                co_relate(in_name, co_related, existing_cross_items_of_variable)

        # Figure out which derivative output channel has been silenced (constant handled)
        # 2 ways of silenced: it is not differentiable, it is zeroed.
        def graph_name_of_output_channel(out_channel: Tuple):
            assert len(out_channel) in {1, 2, 3}
            if len(out_channel) == 1:
                return self_output_variables[out_channel[0]].nick_name
            else:
                return get_graph_derivative_name(
                    self_output_variables[out_channel[0]].nick_name,
                    [self_input_variables[in_idx].nick_name for in_idx in out_channel[1:]])

        required_channels = outer_full_context.required_output_channels()

        lines_to_be_inserted_to_bracket_begin = [Const1005.indent + "// Link outputs by pointer."]
        for channel in required_channels:
            fully_differentiable = \
                self_output_variables[channel[0]].is_differentiable() and all(
                    [self_input_variables[in_idx].is_differentiable() for in_idx in channel[1:]])

            if not fully_differentiable:
                # assert False, "The context required channel is not differentiable"
                result.constant_output_channels[channel] = 0
            else:
                graph_var_name = graph_name_of_output_channel(channel)
                if manager.is_constant(graph_var_name):
                    result.constant_output_channels[channel] = manager.get_constant_value(graph_var_name)
                else:
                    field_name = outer_full_context.output_channel_name(channel)
                    field_type = outer_full_context.output_channel_type(channel)
                    lines_to_be_inserted_to_bracket_begin.append(
                        Const1005.indent + field_type + "* " + Const1005.graph_output_pointer_prefix + field_name +
                        " = &" + field_name + ";")
                    append_line("*" + Const1005.graph_output_pointer_prefix + field_name + " = " + graph_var_name + ";",
                                indent_num=1)
        append_line("}")
        insert_lines_to_bracket_begin(lines_to_be_inserted_to_bracket_begin)

        return result

    def create_graph_function(self,
                              input_variables: List[Variable],
                              output_variables: List[Variable]):
        """
        :param input_variables: input variables in this graph to be treated as function input
        :param output_variables: variables in this graph to be treated as function output
        :return:
        """
        return GraphFunction(graph=self, graph_input_variables=input_variables, graph_output_variables=output_variables)


class GraphFunction(FunctionBase):
    """
    WARNING: this class is dynamical such that .graph is not owned.
    Use WrappedFunction to snapshot the graph function.
    """

    #  TODO(): implement with :
    #  1, checking no - loop dependency
    #  2, assert each outer graph statefuls are not passed as config to this graph.

    def __init__(self,
                 graph: "Graph",
                 graph_input_variables: List[Variable],
                 graph_output_variables: List[Variable]):
        input_spec = [variable.var_type for variable in graph_input_variables]
        output_spec = [variable.var_type for variable in graph_output_variables]
        assert all([variable.graph is graph for variable in graph_input_variables]), \
            "each graph input variable must belongs to the graph"
        assert all([variable.graph is graph for variable in graph_output_variables]), \
            "each graph output variable must belongs to the graph"

        super(GraphFunction, self).__init__(input_spec, output_spec, dependencies=graph.evaluate_all_dependencies())

        self.graph = graph
        self.graph_input_variables = graph_input_variables
        self.graph_output_variables = graph_output_variables

        self.options = list(self.supported_options)
        self.options.sort(key=lambda option: option.to_string())

        self._definitions_per_option = []
        for option in self.options:
            self._definitions_per_option.append(self.graph.get_definition(option))

    def get_definition(self, option: Option) -> DefinitionResult:
        """
        :return:
        """
        assert option in self.supported_options
        return self._definitions_per_option[self.options.index(option)]

    def print_call(self, full_context: FullContext) -> CallResult:
        return self.graph.print_call(full_context, self.graph_input_variables, self.graph_output_variables)
