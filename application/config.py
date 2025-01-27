import json
import os

class Config:
    def __init__(self, file_path):
        """
        Initialize the Config object.

        :param file_path: Path the the JSON configuration file.
        """

        self.file_path = file_path
        self.config = self.load_config()

    def load_config(self):
        """Load the configuration file."""
        if not os.path.exists(self.file_path):
            # Create an empty JSON file if it doesn't exist
            with open(self.file_path, 'w') as file:
                json.dump({}, file)
            return {}
        # Attempt to load the file
        try:
            with open(self.file_path, 'r') as file:
                content = file.read().strip()
                return json.loads(content) if content else {}
        except json.JSONDecodeError:
            # Reset the file if corrupted
            print(f"Warning: {self.file_path} is not valid JSON. Resetting.")
            with open(self.file_path, 'w') as file:
                json.dump({}, file)
            return {}

    def save_config(self):
        """Save the configuration file."""
        with open(self.file_path, 'w') as configFile:
            json.dump(self.config, configFile, indent=4)
        
    def _navigate_to_key(self, key_path, create_missing=False):
        """
        Navigate to the nested dictionary where the key resides.

        :param key_path: List of keys (split bt '.')
        :param create_missing: if True, creates missing keys  as empty dictionaries
        :return: The dictionary containing the final key, or None if the path doesn't exist.
        """
        data = self.config
        for key in key_path[:-1]:
            if key not in data:
                if create_missing:
                    data[key] = {}
                else:
                    return None
            data = data[key]
        return data

    def get_value(self, key) -> str:
        """Get the value of a key in the configuration.

        :param key: Key path seperated by dots (e.g., "section.sub1.sub2.value")
        :return: Value of key, or None if not found
        """
        key_path = key.split('.')
        data = self._navigate_to_key(key_path)
        if data and key_path[-1] in data:
            return str(data[key_path[-1]])
        raise ValueError(f"ERROR: Invalid settings for {key}.")

    def set_value(self, key, value) -> None:
        """
        Set or update the value of a key in the configuration.

        :param key: Key path seperated by dots (e.g., "section.sub1.sub2.value")
        :param value: The value to set
        """
        key_path = key.split('.')
        data = self._navigate_to_key(key_path, create_missing=True)
        if data is not None:
            data[key_path[-1]] = value
            self.save_config()

    def remove_key(self, key) -> None:
        """
        Remove a key from the configuration.

        :param key: Key path seperated by dots (e.g., "section.sub1.sub2.val")
        """
        key_path = key.split('.')
        data = self._navigate_to_key(key_path)
        if data and key_path[-1] in data:
            del data[key_path[-1]]
            self.save_config()

    def remove_section(self, key) -> None:
        """
        Remove an entire section (nested dictionary).

        :param key: Key path seperated by dots (e.g., "section.sub1.sub2.val")
        """
        key_path = key.split('.')
        parent_data = self._navigate_to_key(key_path)
        if parent_data and key_path[-1] in parent_data:
            del parent_data[key_path[-1]]
            self.save_config()


if __name__ == "__main__":
    config = Config("settings.json")

    config.set_value("assets.section.val", "test")
    config.set_value("environment.section.val", "test")

    config.get_value("environment.geography.filepath")
