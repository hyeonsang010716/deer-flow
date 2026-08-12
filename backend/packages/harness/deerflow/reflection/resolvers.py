from importlib import import_module

MODULE_TO_PACKAGE_HINTS = {
    "langchain_google_genai": "langchain-google-genai",
    "langchain_anthropic": "langchain-anthropic",
    "langchain_openai": "langchain-openai",
    "langchain_deepseek": "langchain-deepseek",
}


def _build_missing_dependency_hint(module_path: str, err: ImportError) -> str:
    """module import 실패 시 바로 조치 가능한 힌트를 만든다."""
    module_root = module_path.split(".", 1)[0]
    missing_module = getattr(err, "name", None) or module_root

    # 알려진 integration은 provider 패키지 힌트를 우선한다. import 에러가 전이 의존성(예:
    # `google`) 때문에 발생한 경우에도 마찬가지다.
    package_name = MODULE_TO_PACKAGE_HINTS.get(module_root)
    if package_name is None:
        package_name = MODULE_TO_PACKAGE_HINTS.get(missing_module, missing_module.replace("_", "-"))

    return f"Missing dependency '{missing_module}'. Install it with `uv add {package_name}` (or `pip install {package_name}`), then restart DeerFlow."


def resolve_variable[T](
    variable_path: str,
    expected_type: type[T] | tuple[type, ...] | None = None,
) -> T:
    """경로로부터 변수를 해석한다.

    Args:
        variable_path: 변수 경로(예: "parent_package_name.sub_package_name.module_name:variable_name").
        expected_type: 해석된 변수를 검증할 타입 또는 타입 튜플(선택). 주어지면 isinstance()로
            해당 타입의 인스턴스인지 확인한다.

    Returns:
        해석된 변수.

    Raises:
        ImportError: module 경로가 잘못됐거나 해당 속성이 없을 때.
        ValueError: 해석된 변수가 검증을 통과하지 못했을 때.
    """
    try:
        module_path, variable_name = variable_path.rsplit(":", 1)
    except ValueError as err:
        raise ImportError(f"{variable_path} doesn't look like a variable path. Example: parent_package_name.sub_package_name.module_name:variable_name") from err

    try:
        module = import_module(module_path)
    except ImportError as err:
        module_root = module_path.split(".", 1)[0]
        err_name = getattr(err, "name", None)
        if isinstance(err, ModuleNotFoundError) or err_name == module_root:
            hint = _build_missing_dependency_hint(module_path, err)
            raise ImportError(f"Could not import module {module_path}. {hint}") from err
        # module 누락이 아닌 실패는 원래 ImportError 메시지를 그대로 보존한다.
        raise ImportError(f"Error importing module {module_path}: {err}") from err

    try:
        variable = getattr(module, variable_name)
    except AttributeError as err:
        raise ImportError(f"Module {module_path} does not define a {variable_name} attribute/class") from err

    # 타입 검증
    if expected_type is not None:
        if not isinstance(variable, expected_type):
            type_name = expected_type.__name__ if isinstance(expected_type, type) else " or ".join(t.__name__ for t in expected_type)
            raise ValueError(f"{variable_path} is not an instance of {type_name}, got {type(variable).__name__}")

    return variable


def resolve_class[T](class_path: str, base_class: type[T] | None = None) -> type[T]:
    """module 경로와 class 이름으로부터 class를 해석한다.

    Args:
        class_path: class 경로(예: "langchain_openai:ChatOpenAI").
        base_class: 해석된 class가 이 class의 하위 class인지 확인할 기준 class.

    Returns:
        해석된 class.

    Raises:
        ImportError: module 경로가 잘못됐거나 해당 속성이 없을 때.
        ValueError: 해석된 객체가 class가 아니거나 base_class의 하위 class가 아닐 때.
    """
    model_class = resolve_variable(class_path, expected_type=type)

    if not isinstance(model_class, type):
        raise ValueError(f"{class_path} is not a valid class")

    if base_class is not None and not issubclass(model_class, base_class):
        raise ValueError(f"{class_path} is not a subclass of {base_class.__name__}")

    return model_class
