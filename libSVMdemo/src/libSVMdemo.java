
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Vector;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;

public class libSVMdemo {
	svm_parameter _param;
	svm_problem _prob;
	String _model_file;
	
	protected void loadData(boolean is_training){

		double max_index = 0;
		_prob = new svm_problem();
		Vector<Double> vy = new Vector<Double>();
		Vector<svm_node[]> vx = new Vector<svm_node[]>();
		String tmp;
		int test=1001;
		
		
		FileReader fr = null;
		try {
			if(is_training)
				fr = new FileReader("train_233_init.txt");
			else
				fr = new FileReader("train_233_init.txt");
			BufferedReader br = new BufferedReader(fr);
		
			try {
				int id=0;
				while (br.ready()) {
					tmp=br.readLine();
					
					if(is_training){	//training
						if(id>=test){
							//System.out.print(id+" Loading training data...");
							String[] read_in_data = tmp.split(" ");
							//System.out.println(read_in_data[0]);
							
							if(Double.parseDouble(read_in_data[0])==0.0)
								vy.addElement(1.0);
							else
								vy.addElement(0.0);
							
							svm_node[] x = new svm_node[2];
							x[0] = new svm_node();
							x[0].index = 1;
							x[0].value = Double.parseDouble(read_in_data[1].substring(2));
							x[1] = new svm_node();
							x[1].index = 2;
							x[1].value = Double.parseDouble(read_in_data[2].substring(2));
							max_index = Math.max(max_index, Double.parseDouble(read_in_data[2].substring(2)));
							vx.addElement(x);
						}			
					}else{
						if(id<test){
							//System.out.print(id+" Loading testing data...");
							String[] read_in_data = tmp.split(" ");
							//System.out.println(read_in_data[0]);
							
							if(Double.parseDouble(read_in_data[0])==0.0)
								vy.addElement(1.0);
							else
								vy.addElement(0.0);
							
							svm_node[] x = new svm_node[2];
							x[0] = new svm_node();
							x[0].index = 1;
							x[0].value = Double.parseDouble(read_in_data[1].substring(2));
							x[1] = new svm_node();
							x[1].index = 2;
							x[1].value = Double.parseDouble(read_in_data[2].substring(2));
							max_index = Math.max(max_index, Double.parseDouble(read_in_data[2].substring(2)));
							vx.addElement(x);
						}
					}
										
					id++;
				}
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		try {
			fr.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		if(max_index > 0) _param.gamma = 1.0/max_index;		// 1/num_features	

		_prob.l = vy.size();
		_prob.x = new svm_node[_prob.l][];
		
		for(int i=0;i<_prob.l;i++){
			_prob.x[i] = vx.elementAt(i);
			//System.out.print(_prob.x[i]);
		}
		
		_prob.y = new double[_prob.l]; 
		
		for(int i=0;i<_prob.l;i++){
			_prob.y[i] = vy.elementAt(i);
			//System.out.print(_prob.y[i]);
		}
		
		System.out.println(_prob.l+" Done!!");
	}
	
	protected void training(){
		loadData(true);
		
		System.out.println("Training...");
		_model_file = "svm_model.txt";
			
		try{
			/*for(int i=0;i<_prob.l;i++){
				System.out.println(i+": "+_prob.x[i][0].value+" "+_prob.x[i][1].value+ " "+_prob.y[i]);
			}*/
			svm_model model = svm.svm_train(_prob, _param);
			System.out.println("Done!!"+_param.gamma);
			svm.svm_save_model(_model_file, model);
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	
	protected void testing(){
		loadData(false);
		
		System.out.println("Testing...");
		svm_model model;
		int correct = 0, total = 0;
		try {
			model = svm.svm_load_model(_model_file);
			
			FileWriter fw = new FileWriter("SVM.txt");       
	       

			for(int i=0;i<_prob.l;i++){
				double v;
				svm_node[] x = _prob.x[i];
				v = svm.svm_predict(model, x);
				total++;
				
				//975: 0.461111,0.586111,Y predict_result:0.0 correction:Y
				fw.write(i+": "+_prob.x[i][0].value+","+_prob.x[i][1].value+",");
				if(_prob.y[i]==1.0)
					fw.write("N");
				else
					fw.write("Y");
				fw.write(" predict_result:"+v);
				
				if(v == _prob.y[i]){
					correct++;
					fw.write(" correction:Y\r\n");
				}
				else
					fw.write(" correction:N\r\n");
			}
			
			double accuracy = (double)correct/total;
			System.out.println("Accuracy = "+accuracy+"% ("+correct+"/"+total+")");
			fw.write("accuracy: "+accuracy);
			fw.flush();
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	libSVMdemo(){
		// default values
		_param = new svm_parameter();
		
		_param.svm_type = svm_parameter.C_SVC;
		_param.kernel_type = svm_parameter.LINEAR;
		_param.degree = 3;
		_param.gamma = 0;		// 1/num_features
		_param.coef0 = 0;
		_param.nu = 0.5;
		_param.cache_size = 100;
		_param.C = 1;
		_param.eps = 1e-3;
		_param.p = 0.1;
		_param.shrinking = 1;
		_param.probability = 0;
		_param.nr_weight = 0;
		_param.weight_label = new int[0];
		_param.weight = new double[0];
		
		training();
		testing();
	}
	
	public static void main(String[] args) {
		libSVMdemo ld = new libSVMdemo();
	}
}
