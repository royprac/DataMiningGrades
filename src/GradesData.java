import  java.io.BufferedReader ; 
	import  java.io.FileNotFoundException ; 
	import  java.io.FileReader ; 
	import  weka.classifiers.Classifier ; 
	import  weka.classifiers.Evaluation ; 
	import  weka.classifiers.evaluation.NominalPrediction ; 
	import  weka.classifiers.rules.DecisionTable ; 
	import  weka.classifiers.rules.PART ; 
	import  weka.classifiers.trees.DecisionStump ; 
	import  weka.classifiers.trees.J48 ; 
	import  weka.core.FastVector ; 
	import  weka.core.Instances ;
	
public class GradesData {
	
		public static BufferedReader readDataFile(String filename) {
			BufferedReader inputReader = null;
	 
			try {
				inputReader = new BufferedReader(new FileReader(filename));
			} catch (FileNotFoundException ex) {
				System.err.println("Archivo no Encontrado: " + filename);
			}
	 
			return inputReader;
		}
	 
		public static Evaluation classify(Classifier model,
				Instances trainingSet, Instances testingSet) throws Exception {
			Evaluation evaluation = new Evaluation(trainingSet);
	 
			model.buildClassifier(trainingSet);
			evaluation.evaluateModel(model, testingSet);
	 
			return evaluation;
		}
	 
		public static double calculateAccuracy(FastVector predictions) {
			double correct = 0;
	 
			for (int i = 0; i < predictions.size(); i++) {
				NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
				if (np.predicted() == np.actual()) {
					correct++;
				}
			}
	 
			return 100 * correct / predictions.size();
		}
	 
		public static Instances[][] crossValidationSplit(Instances data, int numberOfFolds) {
			Instances[][] split = new Instances[2][numberOfFolds];
	 
			for (int i = 0; i < numberOfFolds; i++) {
				split[0][i] = data.trainCV(numberOfFolds, i);
				split[1][i] = data.testCV(numberOfFolds, i);
			}
	 
			return split;
		}
	 
		public static void main(String[] args) throws Exception {
			BufferedReader datafile = readDataFile("weather.txt");
	 
			Instances data = new Instances(datafile);
			data.setClassIndex(data.numAttributes() - 1);
	 
			// Hacer 10-split de validación cruzada 
			Instances[][] split = crossValidationSplit(data, 10);
	 
			// División separada en formación y las pruebas arrays 
			Instances[] trainingSplits = split[0];
			Instances[] testingSplits = split[1];
	 
			// Utilizar un conjunto de clasificadores
			Classifier[] models = { 
					new J48(), // Arbol de decisión 
					new PART(), 
					new DecisionTable(),// mesa decisión mayoritaria clasificador 
					new DecisionStump() // árbol de decisiones de un nivel
			};
	 
			// Run para cada modelo 
			for (int j = 0; j < models.length; j++) {
	 
				// Recoge todos los grupo de predicciones para el modelo actual en una FastVector 
				FastVector predictions = new FastVector();
	 
				// Para cada entrenamiento-prueba de par partido, entrenar y probar el clasificador 
				for (int i = 0; i < trainingSplits.length; i++) {
					Evaluation validation = classify(models[j], trainingSplits[i], testingSplits[i]);
	 
					predictions.appendElements(validation.predictions());
	 
					// Descomentar para ver el resumen de cada par de entrenamiento-prueba.
					// FULL PRUEBA ;) 
					// System.out.println(models[j].toString());
				}
	 
				// Calcular precisión global de clasificador actual en todas las divisiones 
				double accuracy = calculateAccuracy(predictions);
	 
				// Escriba el nombre y la precisión del clasificador actual en el compilador. 
				System.out.println("Precisión de " + models[j].getClass().getSimpleName() + ": "
						+ String.format("%.2f%%", accuracy)
						+ "\n---------------------------------");
			}
	 
		}
	}
