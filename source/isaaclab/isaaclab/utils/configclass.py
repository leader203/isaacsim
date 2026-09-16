# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ============================================================
# configclass — Python dataclass 的增强包装器
#
# 解决的问题（标准 dataclass 的两大痛点）：
#
#   1. 强制类型注解：标准 dataclass 要求所有字段必须有类型注解，
#      否则 Python 不将其视为 dataclass 字段，__dict__ 中也不会出现该字段。
#      configclass 自动从默认值推断类型注解，因此可以省略注解。
#
#   2. 可变默认值：标准 dataclass 禁止直接将 list/dict/类实例 作为默认值，
#      必须写 field(default_factory=...)，否则会 ValueError。
#      configclass 自动将任何可变默认值包装为 field(default_factory=...)。
#
# 额外能力：
#   - 每次实例化时对所有可变成员做 deepcopy，彻底隔离不同实例的状态。
#   - 注入 to_dict / from_dict / replace / copy / validate 五个工具方法。
#
# 处理流程（configclass 装饰器执行顺序）：
#   1. _add_annotation_types  — 从 MRO 收集注解，为未注解字段推断类型
#   2. _process_mutable_types — 将可变默认值包装成 field(default_factory=...)
#   3. 注入 __post_init__    — 首次调用用户 __post_init__，再 deepcopy 所有成员
#   4. 注入工具方法           — to_dict / from_dict / replace / copy / validate
#   5. dataclass(cls)        — 最终调用标准 dataclass 完成字段注册
#
# 使用示例见 configclass docstring。
# ============================================================

"""Sub-module that provides a wrapper around the Python 3.7 onwards ``dataclasses`` module."""

import inspect
import types
from collections.abc import Callable
from copy import deepcopy
from dataclasses import MISSING, Field, dataclass, field, replace
from typing import Any, ClassVar

from .dict import class_to_dict, update_class_from_dict

_CONFIGCLASS_METHODS = ["to_dict", "from_dict", "replace", "copy", "validate"]
"""configclass 运行时注入到类上的方法名列表。

这些方法名在 _skippable_class_member 中被跳过，
防止被误当作数据字段处理。
"""

"""
Wrapper around dataclass.
"""


def __dataclass_transform__():
    """Add annotations decorator for PyLance."""
    return lambda a: a


@__dataclass_transform__()
def configclass(cls, **kwargs):
    """Wrapper around `dataclass` functionality to add extra checks and utilities.

    As of Python 3.7, the standard dataclasses have two main issues which makes them non-generic for
    configuration use-cases. These include:

    1. Requiring a type annotation for all its members.
    2. Requiring explicit usage of :meth:`field(default_factory=...)` to reinitialize mutable variables.

    This function provides a decorator that wraps around Python's `dataclass`_ utility to deal with
    the above two issues. It also provides additional helper functions for dictionary <-> class
    conversion and easily copying class instances.

    Usage:

    .. code-block:: python

        from dataclasses import MISSING

        from isaaclab.utils.configclass import configclass


        @configclass
        class ViewerCfg:
            eye: list = [7.5, 7.5, 7.5]  # field missing on purpose
            lookat: list = field(default_factory=[0.0, 0.0, 0.0])


        @configclass
        class EnvCfg:
            num_envs: int = MISSING
            episode_length: int = 2000
            viewer: ViewerCfg = ViewerCfg()


        # create configuration instance
        env_cfg = EnvCfg(num_envs=24)

        # print information as a dictionary
        print(env_cfg.to_dict())

        # create a copy of the configuration
        env_cfg_copy = env_cfg.copy()

        # replace arbitrary fields using keyword arguments
        env_cfg_copy = env_cfg_copy.replace(num_envs=32)

    Args:
        cls: The class to wrap around.
        **kwargs: Additional arguments to pass to :func:`dataclass`.

    Returns:
        The wrapped class.

    .. _dataclass: https://docs.python.org/3/library/dataclasses.html
    """
    # 步骤 1：为所有未注解的成员自动添加类型注解
    # 遍历 MRO（方法解析顺序），从父类到子类依次收集注解，子类覆盖父类同名注解。
    # 对于没有注解的字段，从其默认值 type(value) 推断类型。
    _add_annotation_types(cls)

    # 步骤 2：将所有可变默认值包装为 field(default_factory=...)
    # 这样每次 dataclass __init__ 被调用时，都会通过 factory 创建新对象，
    # 而不是共享同一个可变对象（list/dict/类实例共享会导致状态污染）。
    _process_mutable_types(cls)

    # 步骤 3：注入自定义 __post_init__，在实例化后 deepcopy 所有可变成员
    # 如果用户已定义 __post_init__，则将其与 _custom_post_init 合并：
    #   先执行用户逻辑（如设置派生字段），再执行 deepcopy（隔离状态）。
    if hasattr(cls, "__post_init__"):
        setattr(cls, "__post_init__", _combined_function(cls.__post_init__, _custom_post_init))
    else:
        setattr(cls, "__post_init__", _custom_post_init)

    # 步骤 4：注入五个工具方法
    #   to_dict   — 递归将配置对象转为嵌套字典（用于序列化/打印）
    #   from_dict — 从嵌套字典原地更新配置对象的字段值
    #   replace   — 返回新对象，指定字段替换为新值（类似 dataclasses.replace）
    #   copy      — 返回完整副本（所有字段不变的新实例）
    #   validate  — 递归检查是否有 MISSING 字段，有则抛 TypeError
    setattr(cls, "to_dict", _class_to_dict)
    setattr(cls, "from_dict", _update_class_from_dict)
    setattr(cls, "replace", _replace_class_with_kwargs)
    setattr(cls, "copy", _copy_class)
    setattr(cls, "validate", _validate)

    # 步骤 5：最终调用标准 dataclass(cls)，完成 __init__ / __repr__ 等方法生成
    cls = dataclass(cls, **kwargs)

    # 返回增强后的类
    return cls


"""
Dictionary <-> Class operations.

These are redefined here to add new docstrings.
"""


def _class_to_dict(obj: object) -> dict[str, Any]:
    """Convert an object into dictionary recursively.

    Args:
        obj: The object to convert.

    Returns:
        Converted dictionary mapping.
    """
    return class_to_dict(obj)


def _update_class_from_dict(obj, data: dict[str, Any]) -> None:
    """Reads a dictionary and sets object variables recursively.

    This function performs in-place update of the class member attributes.

    Args:
        obj: The object to update.
        data: Input (nested) dictionary to update from.

    Raises:
        TypeError: When input is not a dictionary.
        ValueError: When dictionary has a value that does not match default config type.
        KeyError: When dictionary has a key that does not exist in the default config type.
    """
    update_class_from_dict(obj, data, _ns="")


def _replace_class_with_kwargs(obj: object, **kwargs) -> object:
    """Return a new object replacing specified fields with new values.

    This is especially useful for frozen classes.  Example usage:

    .. code-block:: python

        @configclass(frozen=True)
        class C:
            x: int
            y: int


        c = C(1, 2)
        c1 = c.replace(x=3)
        assert c1.x == 3 and c1.y == 2

    Args:
        obj: The object to replace.
        **kwargs: The fields to replace and their new values.

    Returns:
        The new object.
    """
    return replace(obj, **kwargs)


def _copy_class(obj: object) -> object:
    """Return a new object with the same fields as the original."""
    # dataclasses.replace(obj) 不传额外参数 → 所有字段保持原值，返回新实例
    return replace(obj)


"""
Private helper functions.
"""


def _add_annotation_types(cls):
    """Add annotations to all elements in the dataclass.

    By definition in Python, a field is defined as a class variable that has a type annotation.

    In case type annotations are not provided, dataclass ignores those members when :func:`__dict__()` is called.
    This function adds these annotations to the class variable to prevent any issues in case the user forgets to
    specify the type annotation.

    This makes the following a feasible operation:

    @dataclass
    class State:
        pos = (0.0, 0.0, 0.0)
           ^^
           If the function is NOT used, the following type-error is returned:
           TypeError: 'pos' is a field but has no type annotation

    实现原理（MRO 遍历）：
      reversed(cls.__mro__) 从最顶层父类（object 除外）到当前类依次遍历，
      子类的注解会覆盖父类同名注解（hints.update），保证继承语义正确。

      对于每个成员：
        - 已有注解 → 直接使用（hints 中已存在，_skippable_class_member 跳过）
        - 无注解 + 不是 type 本身 → 用 type(value) 推断类型并加入 hints
        - 无注解 + 是 type（如嵌套 configclass 类本身当默认值）→ 加注解 "type[ClassName]"
        - 值为 MISSING → 无法推断类型，抛 TypeError 提示用户补注解
    """
    # hints 字典：key=字段名, value=类型注解（从 MRO 层层累积）
    hints = {}

    # 从 MRO 最顶层（最基础的父类）到当前类依次遍历
    # reversed 保证子类注解在最后写入，覆盖父类同名注解
    for base in reversed(cls.__mro__):
        # 跳过 object（内置基类，无业务字段）
        if base is object:
            continue

        # 直接把该基类声明的所有注解合并进 hints
        ann = base.__dict__.get("__annotations__", {})
        hints.update(ann)

        # 遍历该基类的所有成员（保持声明顺序，不用 dir()，因为 dir() 按字母排序）
        for key in base.__dict__:
            value = getattr(base, key)

            # 跳过：dunder / 已注入方法 / 已有注解 / 函数 / property 等（详见 _skippable_class_member）
            if _skippable_class_member(key, value, hints):
                continue

            # 只处理「非 type 本身」的成员（普通字段值）
            if not isinstance(value, type):
                if key not in hints:
                    # MISSING 无法推断类型，必须让用户显式注解
                    if value is MISSING:
                        raise TypeError(
                            f"Missing type annotation for '{key}' in class '{cls.__name__}'."
                            " Please add a type annotation or set a default value."
                        )
                    # 从默认值推断类型（int/float/list/dict/自定义类 等）
                    hints[key] = type(value)
            elif key != value.__name__:
                # 成员是 type 本身（例如 MySubCfg = MySubCfg），且字段名与类名不同
                # 说明是「嵌套配置类」的默认值引用，注解为 "type[ClassName]"
                # Python 3.10+ 类型注解存字符串形式
                hints[key] = f"type[{value.__name__}]"

    # 将累积的 hints 写入当前类的 __annotations__
    # 注意：必须先取 cls.__dict__.get("__annotations__", {}) 而非 cls.__annotations__，
    # 因为后者会把父类注解也包含进来（Python 继承机制），会导致重复。
    cls.__annotations__ = cls.__dict__.get("__annotations__", {})
    cls.__annotations__ = hints


def _validate(obj: object, prefix: str = "") -> list[str]:
    """Check the validity of configclass object.

    This function checks if the object is a valid configclass object. A valid configclass object contains no MISSING
    entries.

    Args:
        obj: The object to check.
        prefix: The prefix to add to the missing fields. Defaults to ''.

    Returns:
        A list of missing fields.

    Raises:
        TypeError: When the object is not a valid configuration object.

    实现说明：
      递归遍历对象的所有字段，直至叶子节点。
      逻辑分支：
        1. obj 是 MISSING 本身          → 直接将当前路径（prefix）加入缺失列表
        2. obj 是 list / tuple           → 递归检查每个元素，路径追加 [index]
        3. obj 是 dict                   → 递归检查每个值，路径追加 .key
        4. obj 有 __dict__（类实例）     → 递归检查实例字典，路径追加 .key
        5. 其他（int/float/str/None 等） → 叶子节点，无需递归，直接返回空列表

      prefix 为空字符串时表示顶层调用：
        - 收集到任何缺失字段后，在此处（且只在此处）抛出 TypeError，避免重复报错。
    """
    missing_fields = []

    # 特殊豁免：MeshConverterCfg 允许有 MISSING 字段（历史遗留）
    if type(obj).__name__ == "MeshConverterCfg":
        return missing_fields

    # 分支 1：obj 本身就是 MISSING → 当前路径缺失
    if type(obj) is type(MISSING):
        missing_fields.append(prefix)
        return missing_fields
    # 分支 2：列表 / 元组 → 按下标递归
    elif isinstance(obj, (list, tuple)):
        for index, item in enumerate(obj):
            current_path = f"{prefix}[{index}]"
            missing_fields.extend(_validate(item, prefix=current_path))
        return missing_fields
    # 分支 3：字典 → 按键递归（非字符串键转为字符串）
    elif isinstance(obj, dict):
        # 非字符串键转为字符串，确保 validate 路径可读
        if any(not isinstance(key, str) for key in obj.keys()):
            obj_dict = {str(key): value for key, value in obj.items()}
        else:
            obj_dict = obj
    # 分支 4：类实例（有 __dict__）→ 按实例字典递归
    elif hasattr(obj, "__dict__"):
        obj_dict = obj.__dict__
    # 分支 5：基础类型叶子节点 → 无缺失，直接返回
    else:
        return missing_fields

    # 遍历 obj_dict 的所有键值对（dict 或实例字典）
    for key, value in obj_dict.items():
        # 跳过 Python 内部属性（__class__, __dict__ 等）
        if key.startswith("__"):
            continue
        # 构建递归路径（根路径不加点前缀）
        current_path = f"{prefix}.{key}" if prefix else key
        missing_fields.extend(_validate(value, prefix=current_path))

    # 只在顶层调用（prefix == ""）时统一抛出 TypeError，避免深层递归多次报错
    if prefix == "" and missing_fields:
        formatted_message = "\n".join(f"  - {field}" for field in missing_fields)
        raise TypeError(
            f"Missing values detected in object {obj.__class__.__name__} for the following"
            f" fields:\n{formatted_message}\n"
        )
    return missing_fields


def _process_mutable_types(cls):
    """Initialize all mutable elements through :obj:`dataclasses.Field` to avoid unnecessary complaints.

    By default, dataclass requires usage of :obj:`field(default_factory=...)` to reinitialize mutable objects
    every time a new class instance is created. If a member has a mutable type and it is created without
    specifying the `field(default_factory=...)`, then Python throws an error requiring the usage of `default_factory`.

    Additionally, Python only explicitly checks for field specification when the type is a list, set or dict.
    This misses the use-case where the type is class itself. Thus, the code silently carries a bug with it which
    can lead to undesirable effects.

    This function deals with this issue

    This makes the following a feasible operation:

    @dataclass
    class State:
        pos: list = [0.0, 0.0, 0.0]
           ^^
           If the function is NOT used, the following value-error is returned:
           ValueError: mutable default <class 'list'> for field pos is not allowed: use default_factory

    实现原理：
      1. 从 MRO 收集所有已知成员到 class_members 字典（子类覆盖父类同名成员）。
         同时也收集父类中已经是 dataclass Field 的成员（因为它们在 __dict__ 里
         已被替换为 Field 对象，不在普通成员中出现）。
      2. 检查 class_members 数量与注解数量是否一致（调试保障）。
      3. 遍历所有注解，对每个字段：
           - ClassVar 类型 → 跳过（不是 dataclass 字段）
           - 已经是 Field 对象 → 直接 setattr（保留用户手写的 field(...)）
           - 其他可变类型 → 用 field(default_factory=_return_f(value)) 包装
    """
    # 只看当前类新增的注解（不含父类）
    ann = cls.__dict__.get("__annotations__", {})

    # 从 MRO（子类优先，reversed 使父类先入，子类覆盖）聚合所有成员
    class_members = {}
    for base in reversed(cls.__mro__):
        if base is object:
            continue
        # 收集普通类成员（未成为 dataclass Field 的）
        for key in base.__dict__:
            f = getattr(base, key)
            if _skippable_class_member(key, f):
                continue
            # 只存「非 type 本身」或「在注解中有记录」的成员
            if not isinstance(f, type) or key in ann:
                class_members[key] = f
        # 收集已成为 dataclass Field 的成员
        # 当父类已经是 dataclass 时，其字段在 __dict__ 中只存为 Field 对象，
        # 不会出现在普通 __dict__ 遍历中，需要从 __dataclass_fields__ 补取。
        for key, f in base.__dict__.get("__dataclass_fields__", {}).items():
            if not isinstance(f, type):
                class_members[key] = f

    # 调试保障：注解数量与成员数量必须一致
    if len(class_members) != len(ann):
        raise ValueError(
            f"In class '{cls.__name__}', number of annotations ({len(ann)}) does not match number of class members"
            f" ({len(class_members)}). Please check that all class members have type annotations and/or a default"
            " value. If you don't want to specify a default value, please use the literal `dataclasses.MISSING`."
        )

    # 遍历所有注解，对可变默认值进行包装
    for key in ann:
        value = class_members.get(key, MISSING)

        # 跳过 ClassVar（如 ClassVar[int]），它不是 dataclass 字段
        origin = getattr(ann[key], "__origin__", None)
        if origin is ClassVar:
            continue

        if isinstance(value, Field):
            # 用户已手动使用 field(...)，直接原样设回（保留用户意图）
            setattr(cls, key, value)
        elif not isinstance(value, type):
            # 普通可变值（list / dict / 类实例等）→ 包装为 field(default_factory=...)
            # _return_f(value) 返回一个闭包，每次调用都 deepcopy(value)，
            # 确保不同实例拿到独立副本。
            value = field(default_factory=_return_f(value))
            setattr(cls, key, value)


def _custom_post_init(obj):
    """Deepcopy all elements to avoid shared memory issues for mutable objects in dataclasses initialization.

    This function is called explicitly instead of as a part of :func:`_process_mutable_types()` to prevent mapping
    proxy type i.e. a read only proxy for mapping objects. The error is thrown when using hierarchical data-classes
    for configuration.

    实现说明：
      尽管 _process_mutable_types 已通过 default_factory 为每个实例创建独立的初始副本，
      但当配置类嵌套（如 EnvCfg 中有 ViewerCfg），deepcopy 才能保证：
        - 内层配置对象在不同 EnvCfg 实例之间完全独立，互不影响。
        - 防止 mappingproxy（只读代理）类型错误（继承链中类属性的只读视图）。

      遍历 dir(obj)（包含继承链上所有成员），对每个非 dunder、非 callable、非 property 的字段，
      执行 setattr(obj, key, deepcopy(value))，将共享引用替换为私有副本。
    """
    for key in dir(obj):
        # 跳过 Python 内置属性和方法（__class__, __init__ 等）
        if key.startswith("__"):
            continue
        # 获取成员值
        value = getattr(obj, key)
        # 检查该成员是否为 property（通过类的 __dict__ 查找）
        ann = obj.__class__.__dict__.get(key)
        # 只对「可变数据字段」做 deepcopy：跳过 callable（方法）和 property（计算属性）
        if not callable(value) and not isinstance(ann, property):
            setattr(obj, key, deepcopy(value))


def _combined_function(f1: Callable, f2: Callable) -> Callable:
    """Combine two functions into one.

    Args:
        f1: The first function.
        f2: The second function.

    Returns:
        The combined function.

    使用场景：
      当用户在 configclass 子类中定义了 __post_init__ 时，
      我们需要同时执行：
        1. 用户的 __post_init__（如修改派生字段，self.step_dt = self.sim.dt * self.decimation）
        2. _custom_post_init（deepcopy 所有可变成员，隔离实例间状态）
      顺序固定：先用户逻辑，再 deepcopy，确保 deepcopy 的是最终状态。
    """

    def _combined(*args, **kwargs):
        # 先调用用户自定义的 __post_init__（f1）
        f1(*args, **kwargs)
        # 再调用 configclass 注入的 deepcopy 逻辑（f2）
        f2(*args, **kwargs)

    return _combined


"""
Helper functions
"""


def _skippable_class_member(key: str, value: Any, hints: dict | None = None) -> bool:
    """Check if the class member should be skipped in configclass processing.

    The following members are skipped:

    * Dunder members: ``__name__``, ``__module__``, ``__qualname__``, ``__annotations__``, ``__dict__``.
    * Manually-added special class functions: From :obj:`_CONFIGCLASS_METHODS`.
    * Members that are already present in the type annotations.
    * Functions bounded to class object or class.
    * Properties bounded to class object.

    Args:
        key: The class member name.
        value: The class member value.
        hints: The type hints for the class. Defaults to None, in which case, the
            members existence in type hints are not checked.

    Returns:
        True if the class member should be skipped, False otherwise.

    跳过规则详解：
      1. dunder（双下划线前缀）：Python 内置属性，不是业务字段
      2. _CONFIGCLASS_METHODS 中的方法：configclass 自己注入的，不是数据字段
      3. hints 中已存在：父类已注解过，无需重复处理
      4. callable + MethodType：类方法（classmethod / staticmethod 的实例方法形式）
      5. callable + 签名含 self/cls：普通实例方法或类方法
      6. property：计算属性，不存储实际数据

      特殊例外：callable 中包含 "CollisionAPI" 的不跳过
        原因：CollisionAPI 是 Omniverse USD PhysX 的碰撞 API 类，
              尽管 callable，但它是配置字段而非普通方法。
    """
    # 跳过规则 1：dunder 成员（Python 内置属性/方法）
    if key.startswith("__"):
        return True
    # 跳过规则 2：configclass 自己注入的工具方法
    if key in _CONFIGCLASS_METHODS:
        return True
    # 跳过规则 3：已在 hints 中（已注解），避免重复添加
    if hints is not None and key in hints:
        return True
    # 跳过规则 4/5：可调用对象（方法等）
    if callable(value):
        # FIXME: 静态方法在此处仍被视为普通函数（function type），暂未处理
        # 跳过类方法（MethodType 表示已绑定到类的方法）
        if isinstance(value, types.MethodType):
            return True

        # 特殊豁免：CollisionAPI 是 Omniverse 碰撞配置类，虽 callable 但是字段
        if "CollisionAPI" in value.__name__:
            return False

        # 通过函数签名判断：含 self（实例方法）或 cls（类方法）→ 跳过
        signature = inspect.signature(value)
        if "self" in signature.parameters or "cls" in signature.parameters:
            return True

    # 跳过规则 6：property（计算属性）
    if isinstance(value, property):
        return True

    # 不在任何跳过规则中 → 不跳过，需要处理
    return False


def _return_f(f: Any) -> Callable[[], Any]:
    """Returns default factory function for creating mutable/immutable variables.

    This function should be used to create default factory functions for variables.

    Example:

        .. code-block:: python

            value = field(default_factory=_return_f(value))
            setattr(cls, key, value)

    实现原理（闭包工厂）：
      _return_f(f) 返回一个无参函数 _wrap，每次调用 _wrap() 时，
      都返回 f 的深拷贝，确保每个实例拥有独立的可变对象。

      分支处理：
        - f 是 Field 对象（用户已写了 field(...)）：
            若有 default_factory  → 调用 factory（如 list, dict）
            若无 default_factory  → deepcopy(f.default)（使用指定默认值的副本）
        - f 是其他值（list/dict/类实例）：
            直接 deepcopy(f)

      为什么需要闭包（而非直接 deepcopy）？
        _return_f 在类定义时（dataclass 初始化前）被调用一次，
        生成的 _wrap 被存入 Field.default_factory。
        之后每次 dataclass __init__ 都调用 _wrap()，才发生实际 deepcopy，
        保证每个实例独立。
    """

    def _wrap():
        if isinstance(f, Field):
            if f.default_factory is MISSING:
                # Field 有固定默认值 → 深拷贝该默认值
                return deepcopy(f.default)
            else:
                # Field 有 factory → 调用 factory 生成新实例
                return f.default_factory
        else:
            # 普通可变对象 → 深拷贝
            return deepcopy(f)

    return _wrap
