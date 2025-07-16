"""
Attribute Proxy Wrapper for robo-gym environments.

This wrapper ensures that custom attributes from robo-gym environments
are properly exposed through Gymnasium wrappers like OrderEnforcing.
"""

import gymnasium as gym


class AttributeProxyWrapper(gym.Wrapper):
    """
    A wrapper that properly exposes custom attributes through the wrapper chain.
    
    This is necessary because Gymnasium's OrderEnforcing wrapper doesn't expose
    custom attributes like 'ur', 'kill_sim', etc. from the underlying environment.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
    def __getattr__(self, name):
        """
        Forward attribute access to the unwrapped environment if not found in wrapper.
        """
        # First try to get from the wrapper itself
        if hasattr(super(), name):
            return getattr(super(), name)
        
        # Then try the wrapped environment
        if hasattr(self.env, name):
            return getattr(self.env, name)
        
        # Finally try the unwrapped environment
        if hasattr(self.unwrapped, name):
            return getattr(self.unwrapped, name)
        
        # If not found anywhere, raise AttributeError
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def kill_sim(self):
        """Kill simulation by delegating to the underlying environment."""
        if hasattr(self.env, 'kill_sim'):
            return self.env.kill_sim()
        elif hasattr(self.unwrapped, 'kill_sim'):
            return self.unwrapped.kill_sim()
        else:
            # If no kill_sim method is available, do nothing
            pass
