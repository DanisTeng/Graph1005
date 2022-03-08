from t1005_graph import *


def opt_separate_exp_by2(expr: sp.Basic):
    if not isinstance(expr, (sp.Basic,)):
        return expr
    if not expr.args:
        return expr
    # ... TODO(): more optimization
    return expr


class OptimizedCXX11Printer(CXX11CodePrinter):
    # TODO(): More optimization
    pass


class SymPyFunction(FunctionBase):
    def __init__(self, sympy_function, output_dim_override=None, input_dim_override=None):
        # infer the number of sympy function inputs.
        if input_dim_override is not None:
            input_dim = input_dim_override
        else:
            input_dim = sympy_function.__code__.co_argcount
        assert input_dim > 0, "Sympy function should have at least one input"

        output_dim = 1
        if output_dim_override is not None:
            assert type(output_dim_override) is int
            output_dim = output_dim_override
        else:
            # We infer the number of the sympy function outputs.
            # using the following hacky method:
            try:
                a_test_input = [1.0] * input_dim
                test_output = sympy_function(*a_test_input)
                if type(test_output) is tuple or type(test_output) is list:
                    output_dim = len(test_output)
            except:
                assert False, "Failed to infer sympy function output dimension, please specify 'output_dim_override'."

        assert output_dim > 0, "Sympy function should have at least one output"

        input_spec = [''] * input_dim
        output_spec = [''] * output_dim

        super(SymPyFunction, self).__init__(input_spec, output_spec)
        self.sympy_function = sympy_function

    def is_compatible(self, context: Context) -> Tuple[bool, str]:
        res, dbg = super(SymPyFunction, self).is_compatible(context)

        if not res:
            return res, dbg

        if not all([VarType1005.is_numerical_var_type(input_var.var_type) for input_var in context.input_variables]):
            return False, "SympyFunction input variables must be numerical."

        if not all([VarType1005.is_numerical_var_type(output_var.var_type) for output_var in context.output_variables]):
            return False, "SympyFunction output variables must be numerical."

        return True, ""

    def print_call(self, full_context: FullContext) -> CallResult:
        res, dbg = self.is_compatible(full_context.context)
        assert res, dbg

        in_dim = len(full_context.context.input_variables)
        out_dim = len(full_context.context.output_variables)

        result = CallResult()
        # sympy environment
        # zero_order_output_exprs[i] corresponds to output_variable[i]
        # input_symbols[i] corresponds to input_variable[i]
        input_symbols = []
        for input_variable in full_context.context.input_variables:
            if input_variable.type is Variable.TYPE_CONSTANT:
                input_symbols.append(input_variable.value)
            else:
                input_symbols.append(sp.symbols(input_variable.nick_name))

        zero_order_output_exprs = self.sympy_function(*input_symbols)
        if len(full_context.context.output_variables) == 1:
            assert type(zero_order_output_exprs) is not tuple
            zero_order_output_exprs = [zero_order_output_exprs, ]
        else:
            assert type(zero_order_output_exprs) is tuple or type(zero_order_output_exprs) is list
        assert len(zero_order_output_exprs) == out_dim

        # Get the sympy expressions of each output.
        result_exprs = []
        result_names = []
        result_types = []

        output_channels = full_context.required_output_channels()

        for channel in output_channels:
            is_constant_derivative_output = False
            if len(channel) == 1:
                expr = zero_order_output_exprs[channel[0]]
            elif len(channel) == 2:
                expr = zero_order_output_exprs[channel[0]].diff(input_symbols[channel[1]])
                is_constant_derivative_output = expr.is_Number
            elif len(channel) == 3:
                expr = (zero_order_output_exprs[channel[0]].diff(input_symbols[channel[1]])).diff(
                    input_symbols[channel[2]])
                is_constant_derivative_output = expr.is_Number
            else:
                assert False

            res_name = full_context.output_channel_name(channel)
            res_type = full_context.output_channel_type(channel)

            # Arbitrate here/
            if is_constant_derivative_output:
                result.constant_output_channels[channel] = float(expr)
            else:
                result_exprs.append(expr)
                result_names.append(res_name)
                result_types.append(res_type)

        # Calculate the simplified expression.
        replacements, reduced_exprs = sp.cse(result_exprs,
                                             sp.utilities.iterables.numbered_symbols(Const1005.sympy_var_prefix))
        assert type(reduced_exprs) is tuple or type(reduced_exprs) is list
        assert len(result_exprs) == len(reduced_exprs)

        printer = OptimizedCXX11Printer()

        sympy_local_var_type = \
            VarType1005.infer_combined_var_type(
                [input_variable.var_type for input_variable in full_context.context.input_variables])

        assert sympy_local_var_type is not None

        # calculation steps
        result.lines.append("{")
        for replacement in replacements:
            this_line = printer.doprint(replacement[1], sympy_local_var_type + ' ' + str(replacement[0])).replace('\n',
                                                                                                                  '')
            result.lines.append(Const1005.indent + this_line)
        # assign outputs
        for i in range(len(reduced_exprs)):
            this_line = printer.doprint(reduced_exprs[i], result_names[i])
            result.lines.append(Const1005.indent + this_line)
        result.lines.append("}")

        return result


class _SymPyOperatorFunctions:
    sympy_add = SymPyFunction(lambda a, b: a + b)
    sympy_sub = SymPyFunction(lambda a, b: a - b)
    sympy_mul = SymPyFunction(lambda a, b: a * b)
    sympy_pow = SymPyFunction(lambda a, b: a ** b)
    sympy_div = SymPyFunction(lambda a, b: a / b)


def _variable_add(self, other):
    return _SymPyOperatorFunctions.sympy_add(self, other)


def _variable_radd(self, other):
    return _SymPyOperatorFunctions.sympy_add(other, self)


def _variable_sub(self, other):
    return _SymPyOperatorFunctions.sympy_sub(self, other)


def _variable_rsub(self, other):
    return _SymPyOperatorFunctions.sympy_sub(other, self)


def _variable_mul(self, other):
    return _SymPyOperatorFunctions.sympy_mul(self, other)


def _variable_rmul(self, other):
    return _SymPyOperatorFunctions.sympy_mul(other, self)


def _variable_pow(self, other):
    return _SymPyOperatorFunctions.sympy_pow(self, other)


def _variable_rpow(self, other):
    return _SymPyOperatorFunctions.sympy_pow(other, self)


def _variable_truediv(self, other):
    return _SymPyOperatorFunctions.sympy_div(self, other)


def _variable_rtruediv(self, other):
    return _SymPyOperatorFunctions.sympy_div(other, self)


setattr(Variable, '__add__', _variable_add)
setattr(Variable, '__radd__', _variable_radd)
setattr(Variable, '__sub__', _variable_sub)
setattr(Variable, '__rsub__', _variable_rsub)
setattr(Variable, '__mul__', _variable_mul)
setattr(Variable, '__rmul__', _variable_rmul)
setattr(Variable, '__pow__', _variable_pow)
setattr(Variable, '__rpow__', _variable_rpow)
setattr(Variable, '__truediv__', _variable_truediv)
setattr(Variable, '__rtruediv__', _variable_rtruediv)
