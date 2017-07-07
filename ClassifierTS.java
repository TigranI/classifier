package classifier;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import classifier.PrimalSvmTree.TreeModel;

// 
//

public class ClassifierTS {

	int N;
	int dim;
	double[][] X; // training data
	int[] Y;  // +1, -1

	double[][] Xt;  // testing data
	int[] Yt;  // +1, -1

	int Nplus;
	int Nminus;
	int majority;

	int maxPoints;
	int maxDims;

	BinaryModel binaryModel = null;

	public ClassifierTS(double[][] X, int[] Y, double[][] Xt, int[] Yt) {

		this.X = X;
		this.Y = Y;
		this.Xt = Xt;
		this.Yt = Yt;
		N = X.length;
		Nplus = 0;
		Nminus = 0;
		dim = X[0].length;

		maxDims = 10;
		maxPoints = 1000;		

		buildModel();
		testModel();
	}

	public void buildModel() {

		for(int j=0;j<N;j++) {
			if(Y[j] == 1) {
				Nplus++;
			} else {
				Nminus++;
			}
		}

		if(Nplus >= Nminus) {
			majority = 1;
		} else {
			majority = -1;
		}

		MdsData mdsData = null;
		ScaleData scaleData = null;

		if(dim > maxDims) {
			Lmds lmds = new Lmds(X, maxDims);
			X = lmds.getResultSpace();
			mdsData = new MdsData(lmds.getLandmarks(), lmds.getMatrix(), lmds.getMeans());
		}

		dim = X[0].length;
		scaleData = new ScaleData(scale());
		TreeSvmModel treeSvmModel = decompose();
		binaryModel = new BinaryModel(majority, mdsData, scaleData, treeSvmModel);

		//binaryModel.print();
	}

	public void testModel() {

		int Nt = Yt.length;
		TreeSvmModel treeSvmModel = binaryModel.getTreeModel();
		MdsData  mdsData = binaryModel.getMdsData();
		ScaleData scaleData = binaryModel.getScaleData();

		if(mdsData != null) {
			System.out.println("MDS transforming ...");
		}
		if(scaleData != null) {
			System.out.println("Scale transforming ...");
		}

		int majority = binaryModel.getMajority();

		int[] Ysvm = new int[Nt];

		double testError = 0;
		for(int i=0;i<Nt;i++) {

			if(mdsData != null) {
				Xt[i] = mdsData.applyToPoint(Xt[i]);
			}
			if(scaleData != null) {
				Xt[i] = scaleData.applyToPoint(Xt[i]);
			}

			double value = treeSvmModel.applyToPoint(Xt[i]);
			if(value != Double.NEGATIVE_INFINITY) {
				Ysvm[i] = (int) value;
			} else {
				Ysvm[i] = majority;
			}
			if(Ysvm[i] * Yt[i] <= 0) {
				testError += 1;
			}
		}
		testError /= Nt;

		System.out.println("Test error = " + testError);

	}

	public TreeSvmModel decompose() {

		List<Integer> finalPoints = new ArrayList<Integer>();
		Set<Integer> finalSet = new HashSet<Integer>();

		List<Integer> plus = new ArrayList<Integer>();
		List<Integer> minus = new ArrayList<Integer>();

		for(int j=0;j<N;j++) {
			if(Y[j] == 1) {
				plus.add(j);
			} else {
				minus.add(j);
			}
		}

		double plusRatio = Nplus / (double) N;
		//double minusRatio = Nminus / (double) N;

		int totalGroups  = (int) Math.ceil(N / (double) maxPoints);
		int groupSize = Math.min(N, maxPoints);
		int plusSize = (int) (Math.round(plusRatio * groupSize));
		int minusSize = groupSize - plusSize;
		int numGroups = 0;
		int iplus = 0;
		int iminus = 0;

		while(numGroups < totalGroups) {
			List<Integer> subset = new ArrayList<Integer>();
			int numplus = 0;
			int numminus = 0;
			while(numplus < plusSize && iplus < plus.size()) {
				subset.add(plus.get(iplus));
				iplus++;
				numplus++;
			}
			while(numminus < minusSize && iminus < minus.size()) {
				subset.add(minus.get(iminus));
				iminus++;
				numminus++;
			}
			if(subset.size() < groupSize && numGroups == totalGroups - 1 && totalGroups > 1) {
				iplus = 0;
				iminus = 0;
				while(numplus < plusSize && iplus < plus.size()) {
					subset.add(plus.get(iplus));
					iplus++;
					numplus++;
				}
				while(numminus < minusSize && iminus < minus.size()) {
					subset.add(minus.get(iminus));
					iminus++;
					numminus++;
				}
			}

			int M = subset.size();
			double[][] subsetX = new double[M][dim+1];
			int[] subsetY = new int[M];
			int numSV = Math.min(N, (int) (maxPoints / (double) totalGroups));
			for(int i=0;i<M;i++) {
				for(int j=0;j<dim;j++) {
					subsetX[i][j] = X[subset.get(i)][j];
					subsetY[i] = Y[subset.get(i)];
				}
				subsetX[i][dim] = 1.0;  // b for svm
			}
			System.out.println("computing model for group " + numGroups + " out of " + totalGroups + " : " + M + "   " + numSV + "  " + plusRatio + "  " +
					subsetX.length + "  " + subsetY.length);
			PrimalSvmTree svmtree = new PrimalSvmTree(subsetX, subsetY, numSV);
			TreeModel treemodel = svmtree.getModel();

			for(int index : treemodel.getSVs()) {
				finalSet.add(subset.get(index));
			}

			numGroups++;
		}

		// compute final model
		
		int T = Math.min(N, maxPoints);
		int index = 0;
		while(finalSet.size() < T) {
			finalSet.add(index);
			index++;
		}

		for(int pt : finalSet) {
			finalPoints.add(pt);
		}
		int M = finalPoints.size();
		double[][] subsetX = new double[M][dim+1];
		int[] subsetY = new int[M];
		int numSV = 0;
		for(int i=0;i<M;i++) {
			for(int j=0;j<dim;j++) {
				subsetX[i][j] = X[finalPoints.get(i)][j];
				subsetY[i] = Y[finalPoints.get(i)];
			}
			subsetX[i][dim] = 1.0;  // b for svm
		}

		System.out.println("computing final model : " + M);

		PrimalSvmTree svmtree = new PrimalSvmTree(subsetX, subsetY, numSV);
		TreeModel finalTreeModel = svmtree.getModel();                  // full model

		TreeSvmModel treeSvmModel = new TreeSvmModel(finalTreeModel);  // only takes ws

		return treeSvmModel;
	}

	// scales X, returns scaling coefficients for the classification model
	public double[][] scale() {

		double[][] range = new double[dim][2];
		for(int i=0;i<dim;i++) {
			range[i][0] = Double.POSITIVE_INFINITY;
			range[i][1] = Double.NEGATIVE_INFINITY;
		}


		for(int i=0;i<dim;i++) {
			for(int j=0;j<N;j++) {
				range[i][0] = (range[i][0] < X[j][i]) ? range[i][0] : X[j][i];
				range[i][1] = (range[i][1] > X[j][i]) ? range[i][1] : X[j][i];
			}
		}

		double[][] coefs = new double[dim][2];

		for(int i=0;i<dim;i++) {
			double a = range[i][0];
			double b = range[i][1];

			if(b > a) {
				coefs[i][0] = 2.0 / (b - a);
				coefs[i][1] = (a + b)/(a - b);
			}  else {
				coefs[i][0] = 0.0;
				coefs[i][1] = 0.0;
			}
		}

		for(int j=0;j<N;j++) {
			for(int i=0;i<dim;i++) {
				X[j][i] = coefs[i][0] * X[j][i] + coefs[i][1];
			}
		}

		return coefs;
	}

	class BinaryModel {

		int majority;
		MdsData mdsData;
		ScaleData scaleData;
		TreeSvmModel treeSvmModel;

		public BinaryModel(int majority, MdsData mdsData, ScaleData scaleData, TreeSvmModel treeSvmModel) {

			this.majority = majority;
			this.mdsData = mdsData;
			this.scaleData = scaleData;
			this.treeSvmModel = treeSvmModel;
		}

		public int getMajority() {
			return majority;
		}

		public MdsData getMdsData() {
			return mdsData;
		}

		public ScaleData getScaleData() {
			return scaleData;
		}

		public TreeSvmModel getTreeModel() {
			return treeSvmModel;
		}

		public void print() {
			Map<String, double[]> ws = treeSvmModel.getWs();
			for(String address : ws.keySet()) {
				System.out.println(address + " : " + ws.get(address)[0]);
			}
		}
	}

	class MdsData {

		double[][] landmarks;
		double[][] L;
		double[] lmeans;

		public MdsData(double[][] landmarks, double[][] L, double[] lmeans) {
			this.landmarks = landmarks;
			this.L = L;
			this.lmeans = lmeans;
		}

		public double[][] getLandmarks() {
			return landmarks;
		}

		public double[][] getL() {
			return L;
		}

		public double[] getLmeans() {
			return lmeans;
		}

		public double[] applyToPoint(double[] x) {

			int k = landmarks.length;
			double[] z = new double[k];
			for(int j=0;j<k;j++) {
				z[j] = lmeans[j] - sqL2_dist(x, landmarks[j]);
			}
			double[] w = mult(mult(z, L), 0.5);

			return w;
		}

		double sqL2_dist(double[] x, double[] y) {
			double dist = 0;
			for(int i=0;i<x.length;i++)
				dist += (x[i] - y[i]) * (x[i] - y[i]);
			return dist;
			// return Math.sqrt(dist);
		}

		// v * A
		public double[] mult(double[] v, double[][] A) {

			int N = A[0].length;
			int K = v.length;
			double[] w = new double[N];

			for(int i=0;i<N;i++) {
				for(int j=0;j<K;j++) {
					w[i] += A[j][i] * v[j];
				}
			}

			return w;
		}

		// v * A
		public double[] mult(double[] v, double a) {

			int K = v.length;
			double[] w = new double[K];
			for(int j=0;j<K;j++) {
				w[j] = a * v[j];
			}

			return w;
		}
	}

	class ScaleData {

		double[][] coefs;  // coefficients of linear scaling to [-1, 1] in each dimension

		public ScaleData(double[][] coefs) {
			this.coefs = coefs;
		}

		public double[][] getScalingCoefs() {
			return coefs;
		}

		public double[] applyToPoint(double[] x) {

			double[] sx = new double[x.length];
			for(int i=0;i<x.length;i++) {
				sx[i] = coefs[i][0] * x[i] + coefs[i][1];
			}
			return sx;
		}
	}

	class TreeSvmModel {
		Map<String, double[]> ws;

		public TreeSvmModel(TreeModel treemodel) {
			ws = treemodel.getAllW();
		}

		public Map<String, double[]> getWs() {
			return ws;
		}

		public double applyToPoint(double[] x) {

			double value = Double.NEGATIVE_INFINITY;
			String address = "r";
			boolean flag = true;
			while(flag) {
				if(ws.containsKey(address)) {
					double[] w = ws.get(address);
					value = Math.signum(f(w, x));
					if(value >= 0) {
						address += "+";
					} else {
						address += "-";
					}
				}
				else {
					flag = false;
				}
			}
			return value;
		}

		public double f(double[] w, double[] x) {

			double value = dot(w, x);	
			return value;
		}

		public double dot(double[] v, double[] w) {
			double value = 0;
			int len = Math.min(v.length, w.length);
			for(int i=0;i<len;i++) {
				value += v[i] * w[i];
			}
			if(v.length > len) {
				value += v[len];
			}
			if(w.length > len) {
				value += w[len];
			}
			return value;
		}
	}

	public static void main(String[] args) {

		int N = 800;
		int dim = 2;
		double[][] X = new double[N][dim];
		int[] Y = new int[N];

		double[][] Xt = new double[N][dim];
		int[] Yt = new int[N];

		double a = Math.sqrt(Math.PI / 2.0);
		Random rand = new Random(0);
		for(int i=0;i<N;i++) {
			for(int j=0;j<dim;j++) {
				X[i][j] = 2 * a * rand.nextDouble() - a;
			}
			if(X[i][0] * X[i][0] + X[i][1] * X[i][1] < 1) {
				Y[i] = -1;
			}
			else {
				Y[i] = 1;
			}
		}

		for(int i=0;i<N;i++) {
			for(int j=0;j<dim;j++) {
				Xt[i][j] = 2 * a * rand.nextDouble() - a;
			}
			if(Xt[i][0] * Xt[i][0] + Xt[i][1] * Xt[i][1] < 1) {
				Yt[i] = -1;
			}
			else
				Yt[i] = 1;
		}

		ClassifierTS cts = new ClassifierTS(X, Y, Xt, Yt);
	}
}
