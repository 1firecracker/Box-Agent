"""Session-owned proactive memory plugin, registered explicitly by a host.

ACP registers this plugin; delivery still requires configuration and a capable host.
"""

from ..builtins import AgentRunBindingPort, SessionInitializerPort
from ..descriptors import PluginDescriptor, PluginFactoryContext, PluginScope
from ..registries import CapabilityBinding, CapabilityPolicy
from .bindings import UserTurnObserverPort


def _create_plugin(context: PluginFactoryContext):
    from .runtime import ProactivePlugin

    resources = context.dependencies["session.resources"]
    return ProactivePlugin(
        resources, publish_notification=getattr(resources.context.host, "publish_notification", None),
    )


def plugin_bindings() -> tuple[CapabilityBinding, ...]:
    """Extend the runtime schema without changing the built-in capability list."""
    return (CapabilityBinding(UserTurnObserverPort, CapabilityPolicy.MULTI),)


def plugin_descriptors() -> tuple[PluginDescriptor, ...]:
    from .watch import WatchCheckPlugin
    from .agent_mode import ProactiveModePlugin
    return (PluginDescriptor(
        "session.proactive", "1.0.0", (SessionInitializerPort, UserTurnObserverPort), scope=PluginScope.SESSION,
        dependencies=("session.resources", "session.hooks"), context_factory=_create_plugin,
        disposer=lambda plugin: plugin.close(),
    ), PluginDescriptor(
        "session.proactive_watch", "1.0.0", (SessionInitializerPort, AgentRunBindingPort),
        scope=PluginScope.SESSION, dependencies=("session.resources", "session.hooks"),
        context_factory=lambda context: WatchCheckPlugin(context.dependencies["session.resources"]),
        disposer=lambda plugin: plugin.close(),
    ), PluginDescriptor(
        "session.proactive_agent", "1.0.0", (SessionInitializerPort, AgentRunBindingPort),
        scope=PluginScope.SESSION, dependencies=("session.resources", "session.hooks"),
        context_factory=lambda context: ProactiveModePlugin(context.dependencies["session.resources"]),
        disposer=lambda plugin: plugin.close(),
    ))
