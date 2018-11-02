from flask import Flask
from flask import jsonify,request
import pickle
import ModelUtils as ut
from sklearn.cross_validation import train_test_split

algrithm_name = ""
trained_model = ""

app = Flask(__name__)

@app.route("/train",methods=['GET'])
def training_step():
	### Training Step
	global algrithm_name
	algrithm_name = request.args.get('model_name')
	original_data = ut.read_dataset('Pokemon.csv')
	data = ut.pre_processing(original_data)
	X,Y = ut.split_dataset(data)
	x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = .3)
	selected_model = ut.train(algrithm_name,x_train,y_train)
	pickle.dump(selected_model, open(algrithm_name, 'wb'))
	### Evaluation Step 	
	global trained_model
	trained_model = pickle.load(open(algrithm_name, 'rb'))
	accuracy,cm,report = ut.evaluation(trained_model,x_test,y_test)
	return "Accuracy of the Model <b>("+algrithm_name+")</b> is <b>"+str(accuracy)+"</b>"

	
@app.route("/predict",methods=['GET'])
def predict():
	total = int(request.args.get('total'))
	hp = int(request.args.get('hp'))
	attack = int(request.args.get('attack'))
	defense = int(request.args.get('defense'))
	sp_atk = int(request.args.get('sp_atk'))
	sp_def = int(request.args.get('sp_def'))
	speed = int(request.args.get('speed'))
	generation = int(request.args.get('generation'))
	print("algorithm naem is",algrithm_name)
	
	result = trained_model.predict([[total,hp,attack,defense,sp_atk,sp_def,speed,generation]])
	
	if result[0] == 0:
		return "The Result of The Prediction using <b>("+algrithm_name+")</b>  is <b>{Legendary = False}</b>."
	else:
		return "The Result of The Prediction using <b>("+algrithm_name+")</b>  is <b>{Legendary = True}</b>."
	
	
if __name__ == '__main__':
    app.run(port = 9000, debug = True)	