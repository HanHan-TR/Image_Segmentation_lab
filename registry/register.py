class RegisterMeta(type):
    """Metaclasses implement register templates"""
    def __new__(cls, name, bases, attrs):
        # 自动生成注册器实例的存储字典
        attrs['_storage'] = {}
        return super().__new__(cls, name, bases, attrs)


class Register(metaclass=RegisterMeta):
    """base class of register """
    @classmethod
    def register(cls, name=None):
        """Decorator Registration Methods (Automatic/Manual Naming Supported)"""
        def decorator(func):
            key = name or func.__name__
            if key in cls._storage:
                raise KeyError(f"The {key} is already registered in the {cls.__name__} Register !")
            cls._storage[key] = func
            return func
        return decorator

    @classmethod
    def get(cls, name: str):
        """Get the registration item"""
        if name not in cls._storage:
            raise KeyError(f'Cannot find {name} in {cls.__name__} Register !')

        return cls._storage[name]


class RegisterManager:
    """Register Manager"""
    _registries = {}

    @classmethod
    def create_registry(cls, name: str) -> Register:
        """Dynamically generate the register class"""
        if name not in cls._registries:
            registry_class = type(
                f"{name}Register",
                (Register,),
                {'__name__': name}
            )
            cls._registries[name] = registry_class
        return cls._registries[name]
