from flask import Flask  
from ChatBotApi import chat_api  
  
app = Flask(__name__)  
app.register_blueprint(chat_api, url_prefix='/chat')   
 
@app.route("/")  
def hello():  
    return "Hello World!"  
  
if __name__ == "__main__":  
    app.run()  