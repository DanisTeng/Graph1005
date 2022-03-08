from typing import List, Tuple


class Const1005:
    lower_alphabet_chars = [chr(i) for i in range(97, 123)]
    upper_alphabet_chars = [chr(i) for i in range(65, 91)]
    number_chars = [chr(i) for i in range(48, 58)]

    graph_output_pointer_prefix = "OPTR_"
    graph_input_conveyor_prefix = "INCV_"
    wrapper_graph_output_prefix = "wrapper_"
    sympy_var_prefix = 'sympyvar'
    unnamed_graph_var_prefix = 'graph_var_'
    graph_output_prefix = 'output_'
    node_derivative_prefix = 'NODE_'
    graph_derivative_prefix = "GRAPH_"
    graph_constant_prefix = "G_CONSTANT_"
    graph_unused_prefix = "G_UNUSED_"
    input_channel_short = "input"
    built_in_name_sub_strings = [graph_output_pointer_prefix, sympy_var_prefix, unnamed_graph_var_prefix,
                                  graph_output_prefix]
    indent = "  "


class VarType1005:
    numerical_var_type_complexity = {'int': 1,
                                     'float': 2,
                                     'double': 3,
                                     'ArrayXd': 4}

    var_types_not_need_const_reference = ['double', 'int']
    default_var_type = 'double'

    @classmethod
    def infer_combined_var_type(cls, var_type_list: List[str]):

        combined = None

        for var_type in var_type_list:
            if var_type not in cls.numerical_var_type_complexity:
                continue
            if combined is None:
                combined = var_type
            else:
                if cls.numerical_var_type_complexity[var_type] > cls.numerical_var_type_complexity[combined]:
                    combined = var_type

        return combined

    @classmethod
    def is_transferable(cls, from_type, to_type):
        if from_type not in cls.numerical_var_type_complexity:
            return False

        if to_type not in cls.numerical_var_type_complexity:
            return False

        return cls.numerical_var_type_complexity[from_type] <= cls.numerical_var_type_complexity[to_type]

    @classmethod
    def is_numerical_var_type(cls, var_type):
        return var_type in cls.numerical_var_type_complexity

    @classmethod
    def const_reference(cls, var_type: str):
        assert is_valid_cpp_name(var_type)
        if var_type in cls.var_types_not_need_const_reference:
            return var_type
        else:
            return "const " + var_type + "&"



def full_output_channels_with_derivatives(
        in_dim: int,
        out_dim: int,
        enable_1st_order_derivative: bool = False,
        enable_2nd_order_derivative: bool = False) -> List[Tuple]:
    """
    :param in_dim: input dimension
    :param out_dim: output dimension
    :param enable_1st_order_derivative:
    :param enable_2nd_order_derivative:
    :return: tuple of int representing the order and the channel of the output:
    (i,) Out[i].
    (i,j) d_Out[i]_d_In[j]
    (i,j,k) d2_Out[i]_d_In[j]_d_In[k]
    """
    #
    channels = []

    for i in range(out_dim):
        channels.append((i,))

    if enable_1st_order_derivative:
        for i in range(out_dim):
            for j in range(in_dim):
                channels.append((i, j))

    if enable_2nd_order_derivative:
        for i in range(out_dim):
            for j in range(in_dim):
                for k in range(j, in_dim):
                    channels.append((i, j, k))

    return channels


def get_channel_name(output_and_input_names: Tuple):
    """

    :param output_and_input_names: a tuple with length 1,2, or 3
    :return:
    (out,) out
    (out,in1) D_out_D_in1
    (out,in1,in2) D2_out_D_in1_D_in2
    """
    # in case some uses list
    names = tuple(output_and_input_names)

    l = len(names)

    assert all([is_valid_lower_case_cpp_name(name) for name in names]), "_get_channel_name: invalid input name."

    if l == 1:
        return names[0]
    elif l == 2:
        return "D_%s_D_%s" % names
    elif l == 3:
        return "D2_%s_D_%s_D_%s" % names
    else:
        assert False, "_get_channel_name: invalid input length."


def is_valid_cpp_name(cpp_name: str):
    count = 0
    for c in cpp_name:
        if (count > 0 and c in Const1005.number_chars) or \
                c in Const1005.lower_alphabet_chars or \
                c in Const1005.upper_alphabet_chars or \
                c == '_':
            pass
        else:
            return False
        count += 1
    return count > 0


def is_valid_lower_case_cpp_name(cpp_name: str):
    count = 0
    for c in cpp_name:
        if (count > 0 and c in Const1005.number_chars) or \
                c in Const1005.lower_alphabet_chars or \
                c == '_':
            pass
        else:
            return False
        count += 1
    return count > 0


def is_valid_namespace(namespace: str):
    return all([is_valid_cpp_name(sub_name) for sub_name in namespace.split("::")])


def remove_front_spaces(line: str):
    first_non_space = None
    for i in range(len(line)):
        if line[i] != " ":
            first_non_space = i
            break

    if first_non_space is not None:
        return line[first_non_space:]
    else:
        return ""


def remove_end_spaces(line: str):
    last_non_space = None
    for i in range(len(line) - 1, -1, -1):
        if line[i] != " ":
            last_non_space = i
            break
    if last_non_space is not None:
        return line[:last_non_space + 1]
    else:
        return ""

def string_contains(string:str, sub_string:str):
    return string.find(sub_string)>=0

def remove_front_slashes(line: str):
    first_non_slash = None
    for i in range(len(line)):
        if line[i] != "\\" and line[i] != "/":
            first_non_slash = i
            break

    if first_non_slash is not None:
        return line[first_non_slash:]
    else:
        return ""