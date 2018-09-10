from flask import Blueprint 
from flask import request    
import json  
import ChatBotBOWTalk
  
chat_api = Blueprint('chat_api', __name__)  
 
@chat_api.route("/init")  
def inti():
    return ChatBotBOWTalk.response("Hola")

@chat_api.route("/talk", methods=["POST"])  
def talk():
	data = request.get_json()
	return ChatBotBOWTalk.response(data['msg'])