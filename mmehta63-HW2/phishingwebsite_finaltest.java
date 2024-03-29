package opt.test;

import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.backprop.BatchBackPropagationTrainer;
import func.nn.backprop.RPROPUpdateRule;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.*;
import func.nn.activation.*;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.Scanner;

public class phishingwebsite_finaltest {
    private static Instance[] instances = initializeInstances();
    private static Instance[] train_set = Arrays.copyOfRange(instances, 0, 8844);
    private static Instance[] test_set = Arrays.copyOfRange(instances, 8844, 11055);

    private static DataSet set = new DataSet(train_set);

    private static int inputLayer = 100, outputLayer = 1, trainingIterations = 5000;
    // private static int inputLayer = 46, hiddenLayer=50, outputLayer = 1, trainingIterations = 5000;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");



    public static void write_output_to_file(String output_dir, String file_name, String results, boolean final_result) {
        try {
            if (final_result) {
                String augmented_output_dir = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date());
                String full_path = augmented_output_dir + "/" + file_name;
                Path p = Paths.get(full_path);
                if (Files.notExists(p)) {
                    Files.createDirectories(p.getParent());
                }
                PrintWriter pwtr = new PrintWriter(new BufferedWriter(new FileWriter(full_path, true)));
                synchronized (pwtr) {
                    pwtr.println(results);
                    pwtr.close();
                }
            }
            else {
                String full_path = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date()) + "/" + file_name;
                Path p = Paths.get(full_path);
                Files.createDirectories(p.getParent());
                Files.write(p, results.getBytes());
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

    }



    public static void main(String[] args) {

        String final_result = "";


        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                    new int[] {inputLayer, outputLayer});
                    // new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1e3, .4, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(75, 25, 1, nnop[2]);

        int[] iterations = {10,100,500,1000,2500,5000};

        for (int trainingIterations : iterations) {
            results = "";
            for (int i = 0; i < oa.length; i++) {
                double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0, tp = 0, tn = 0, fp = 0, fn = 0;
                double train_error = train(oa[i], networks[i], oaNames[i]); //trainer.train();
                end = System.nanoTime();
                trainingTime = end - start;
                trainingTime /= Math.pow(10, 9);

                Instance optimalInstance = oa[i].getOptimal();
                networks[i].setWeights(optimalInstance.getData());

                // Calculate Training Set Statistics //
                double predicted, actual;
                start = System.nanoTime();
                for (int j = 0; j < train_set.length; j++) {
                    networks[i].setInputValues(train_set[j].getData());
                    networks[i].run();

                    actual = Double.parseDouble(train_set[j].getLabel().toString());
                    predicted = Double.parseDouble(networks[i].getOutputValues().toString());

                    // System.out.println("actual is " + actual);
                    // System.out.println("predicted is " + predicted);

                   predicted = Math.round(predicted);
                    if (actual == predicted){
                        correct++;
                        if (actual == 0.0)
                            tn++;
                        else
                            tp++;
                    }
                    else{
                        incorrect++;
                        if (actual == 0.0)
                            fp++;
                        else
                            fn++;
                    }

                }
                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10, 9);

                results += "\nTrain Results for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nAccuracy: "
                        + df.format(correct*100 / (correct + incorrect)) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nValidation F1 Score: " + df.format(2.0*tp/(2*tp + fn + fp)) + "\n";

                final_result = oaNames[i] + ",train," + trainingIterations + "," + "training accuracy" + "," + df.format(correct*100 / (correct + incorrect))
                        + "," + "training error" + "," + df.format(train_error) + "," + "training time" + "," + df.format(trainingTime) + "," + "testing time" +
                        "," + df.format(testingTime) + "," + "f1 score" + "," + df.format(2.0*tp/(2*tp + fn + fp));
                write_output_to_file("Optimization_Results", "phishing_results.csv", final_result, true);

                // Calculate Test Set Statistics //
                start = System.nanoTime();
                correct = 0;
                incorrect = 0;
                tp = 0;
                tn = 0;
                fp = 0;
                fn = 0;

                for (int j = 0; j < test_set.length; j++) {
                    networks[i].setInputValues(test_set[j].getData());
                    networks[i].run();

                    actual = Double.parseDouble(test_set[j].getLabel().toString());
                    predicted = Double.parseDouble(networks[i].getOutputValues().toString());

                    predicted = Math.round(predicted);
                    if (actual == predicted){
                        correct++;
                        if (actual == 0.0)
                            tn++;
                        else
                            tp++;
                    }
                    else{
                        incorrect++;
                        if (actual == 0.0)
                            fp++;
                        else
                            fn++;
                    }
                }
                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10, 9);

                results += "\nTest Results for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nAccuracy: "
                        + df.format(correct*100 / (correct + incorrect)) + "%\nTesting time: " + df.format(testingTime) + " seconds\nTest F1 Score: " 
                        + df.format(2.0*tp/(2*tp + fn + fp)) + "\n";

                final_result = oaNames[i] + ",test," + trainingIterations + "," + "testing accuracy" + "," + df.format(correct*100 / (correct + incorrect))
                        + "," + "training error" + "," + df.format(train_error) + "," + "training time" + "," + df.format(trainingTime) + "," + "testing time" +
                        "," + df.format(testingTime) + "," + "f1 score" + "," + df.format(2.0*tp/(2*tp + fn + fp));
                write_output_to_file("Optimization_Results", "phishing_results.csv", final_result, true);
            }
            System.out.println("results for iteration: " + trainingIterations + "---------------------------");
            System.out.println(results);
        }
    }

    private static double train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        // System.out.println("\nError results for " + oaName + "\n---------------------------");

        double error = 0;

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double train_error = 0;
            for(int j = 0; j < train_set.length; j++) {
                network.setInputValues(train_set[j].getData());
                network.run();

                Instance output = train_set[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                train_error += measure.value(output, example);
            }

            error += train_error;
            //System.out.println("training error :" + df.format(train_error)+", testing error: "+df.format(test_error));
        }
        error /= ((float)trainingIterations);
        return error/((float)train_set.length);
    }

    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[11055][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("D:/GATECH/MyStuff/GaTech/CS7641/HW2/SampleSol/PhishingWebsitesData_preprocessed.csv")));

            //for each sample
            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[46]; // 16 attributes
                attributes[i][1] = new double[1]; // classification

                // read features
                for(int j = 0; j < 46; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
                //System.out.println(attributes[i][1][0]);

            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }

        return instances;
    }
}