from forge.config import SQLALCHEMY_DATABASE_URI
from configparser import ConfigParser

config = ConfigParser()

config["DATABASE"] = {"fg_db":SQLALCHEMY_DATABASE_URI}

with open('/etc/forgebox.cfg', 'w') as configfile:
    config.write(configfile)