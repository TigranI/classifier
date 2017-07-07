package classifier;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;

import Jama.Matrix;
import Jama.SingularValueDecomposition;

public class Lmds {

	int m;
	int dim;
	int k;
	double[][] X = null;
	double[][] dist = null;	
	double[][] distL = null;	
	double diamX = 0.0;

	int xdim;
	int maxdim;

	double[][] Y = null;

	double[][] landmarks = null;
	double[][] L = null;
	double[] lmeans = null;

	double[][] v;

	public Lmds(double[][] X, int maxdim) {

		this.X = X;
		m = X.length;
		dim = X[0].length;
		this.maxdim = maxdim;
		k = Math.min(m, maxdim + 30);

		dist = new double[k][k];
		distL = new double[k][m];
		lmeans = new double[k];
		
		// first k points are mds landmarks
		landmarks = new double[k][dim];
		
		process();
	}

	public Lmds(int m, int k, int dim) {
		this.m = m;
		this.dim = dim;
		this.k = k;	
	}

	public void getData() {

		X = new double[m][dim];
		dist = new double[k][k];
		distL = new double[k][m];
		lmeans = new double[k];

		double a = 3;
		v = new double[][]{{a,0}, {-a,0}, {0,a}, {0,-a}};
		int len = v.length;
		Random rand = new Random(0);
		for(int i=0;i<m;i++) {
			int j = i % len;
			double r = (a/4.0) * rand.nextDouble();
			double theta = 2 * Math.PI * rand.nextDouble();
			X[i][0] = r * Math.cos(theta);
			X[i][1] = r * Math.sin(theta);
			add(i, j);
		}
	}

	public void process() {

		computeDist();
		mds(dist);
		embed();
	}

	public void add(int i, int j) {
		for(int k=0;k<X[0].length;k++) {
			X[i][k] = X[i][k] + v[j][k];
		}
	}

	public void computeDist() {

		//System.out.println("m,k,dim = " + m + " " + k + " " + dim);

		for(int i=0;i<k;i++) {
			for(int j=0;j<dim;j++) {
				landmarks[i][j] = X[i][j];
			}
		}

		for(int i=0;i<k;i++) {
			for(int j=0;j<m;j++) {
				double dd = L2_dist(X[i], X[j]);
				distL[i][j] = dd;
				if(j < k) {
					dist[i][j] = dd;
					dist[j][i] = dist[i][j];
					lmeans[i] += dd * dd;
					diamX = (diamX >= dist[i][j]) ? diamX : dist[i][j];
				}
			}
		}

		for(int i=0;i<k;i++) {
			lmeans[i] /= k;
		}
	}

	public void embed() {

		for(int i=k;i<m;i++) {
			double[] x = new double[k];
			for(int j=0;j<distL.length;j++) {
				x[j] = lmeans[j] - distL[j][i] * distL[j][i];
			}
			Y[i] = mult(mult(x, L), 0.5);
		}	
	}

	public void mds(double[][] dist) {

		int n = dist.length;
		double[][] dist1 = new double[n][n];
		for(int i=0;i<n;i++) {
			for(int j=0;j<n;j++) {
				dist1[i][j] = -0.5 * dist[i][j] * dist[i][j];
			}
		}
		Matrix A = new Matrix(dist1);
		Matrix I = Matrix.identity(n, n);
		Matrix J = new Matrix(n, n, 1.0/n);
		Matrix H = I.minus(J);
		Matrix B = H.times(A).times(H);
		double[][] b = B.getArray();
		Pair pair = svd(b);

		double eps = Math.pow(10, -16);
		xdim = 0;
		int len = pair.diag.length;
		double maxEigenvalue = pair.diag[0];
		System.out.println("0 : " + maxEigenvalue);
		int T = Math.min(maxdim, len);
		System.out.println("len,maxdim,T = " + len + " " + maxdim + " " + T);
		double[] ratios = new double[T-1];
		double maxratio = 0;
		for(int i=1;i<T;i++) {
			System.out.println(i + " : " + pair.diag[i]);
			if(pair.diag[i] < eps) {
				//break;
			}
			ratios[i-1] = pair.diag[i-1] / pair.diag[i];
			if(maxratio <= ratios[i-1]) {
				maxratio = ratios[i-1];
				xdim = i;
			}
		}
		xdim = Math.min(xdim, maxdim);
		xdim = Math.max(xdim, 2);
		System.out.println("embedding dim = " + xdim);

		Y = new double[m][xdim];
		L = new double[k][xdim];

		for(int i=0;i<k;i++) {
			for(int j=0;j<xdim;j++) {
				Y[i][j] = pair.U[i][j] * Math.sqrt(pair.diag[j]);  
				L[i][j] = pair.U[i][j] / Math.sqrt(pair.diag[j]);  
			}
		}

	}

	public Pair svd(double[][] x) {

		Matrix AM = new Matrix(x);
		SingularValueDecomposition svd = new SingularValueDecomposition(AM);
		Matrix U = svd.getU();
		double[][] u = U.getArray();
		double[] diag = svd.getSingularValues();

		return new Pair(u, diag);
	}

	// getter functions

	public double[][] getResultSpace() {
		return Y;
	}

	public double[][] getMatrix() {
		return L;
	}

	public double[][] getLandmarks() {
		return landmarks;
	}

	public double[] getMeans() {
		return lmeans;
	}

	// helper functions

	double L2_dist(double[] x, double[] y) {
		double dist = 0;
		for(int i=0;i<x.length;i++)
			dist += (x[i] - y[i]) * (x[i] - y[i]);
		return Math.sqrt(dist);
	}

	class Pair {
		double[][] U;
		double[] diag;

		public Pair(double[][] U, double[] diag) {
			this.U = U;
			this.diag = diag;
		}
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

	public void printData() {		

		String outseparator = ",";
		String outf = "/Users/tigran/Desktop/orthanc_data/testdata/mdstest.csv";

		try {

			FileWriter outFile = new FileWriter(outf);
			PrintWriter out = new PrintWriter(outFile);

			for (int i = 0; i < Y.length; i++) {

				for (int j = 0; j < dim; j++) {
					out.print(Y[i][j] + outseparator);
				}
				out.println("1");
			}
			out.close();
		} catch (IOException ex) {
			ex.printStackTrace();
		}

	}

	public static void main(String[] args) {

		Lmds dd = new Lmds(400, 20, 2);
		dd.process();
		dd.printData();
	}

}
