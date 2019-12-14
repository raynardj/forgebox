import os
DATADIR = os.path.expanduser("~/data")

if os.path.isfile("/etc/forgebox.cfg"):
    from configparser import ConfigParser
    config = ConfigParser()
    config.read("/etc/forgebox.cfg")
    print("loading configs from /etc/forgebox.cfg", flush= True)
    SQLALCHEMY_DATABASE_URI = config["DATABASE"]["fg_db"]
else:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(DATADIR, 'forge.db')