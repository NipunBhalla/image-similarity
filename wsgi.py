from application import application

# do some production specific things to the app

application.config['DEBUG'] = False
if __name__ == "__main__":
    application.run()