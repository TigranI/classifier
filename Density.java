package classifier;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Random;

import org.apache.commons.math.complex.Complex;
import org.apache.commons.math.transform.FastFourierTransformer;

import Jama.*;

public class Density {

	double[][] X = null;
	double[][] dens = null;
	double[][] dist = null;	
	double diamX = 0.0;

	double scale;
	int m;
	int dim;
	String inseparator = ",";
	String in, indist;
	String outseparator = ",";
	String out, outdist;

	int[] ar;

	NumberFormat nf = NumberFormat.getInstance();

	public Density(double scale) {

		this.scale = scale;

		//in = "/Users/tigran/Desktop/orthanc_data/testdata/aaaf_test.csv";
		//out = "/Users/tigran/Desktop/orthanc_data/testdata/zzzf_test.csv";

		in = "/Users/tigran/Desktop/orthanc_data/testdata/aaaf.csv";
		out = "/Users/tigran/Desktop/orthanc_data/testdata/aaaf1.csv";

		indist = "/Users/tigran/Desktop/orthanc_data/testdata/gdist.csv";
		outdist = "/Users/tigran/Desktop/orthanc_data/testdata/gdist1.csv";

		nf.format(3);
		nf.setMaximumFractionDigits(3);

	}

	public void getData() {

		// assumes: id,x1,x2,...,density  format, so dim = col - 2
		try {

			BufferedReader input = new BufferedReader(new FileReader(in));
			try {
				String line = null;
				String[] tokens = null;

				int j = 0;
				while ((line = input.readLine()) != null) {
					if(j > 0) {
						tokens = line.split(inseparator);
						//if(j == 0) {
						dim = tokens.length - 2;  
						//}	
					}
					j++;
				}
				m = j-1;			
				System.out.println("total num pts, dims = " + m + ", " + dim);
			} finally {
				input.close();
			}
		} catch (IOException ex) {
			ex.printStackTrace();
		}

		dim = 2;
		m = 500;
		X = new double[m][dim];
		dist = new double[m][m];
		dens = new double[m][2];  

		// now read data;

		try {

			BufferedReader input = new BufferedReader(new FileReader(in));
			try {
				String line = null;
				String[] tokens = null;

				int j = 0;
				int c = 0;
				while ((line = input.readLine()) != null) {
					if(j > 0) {
						if(j % 2 == 0 || c >= m) {
							j++;
							continue;
						}
						tokens = line.split(inseparator);
						int len = tokens.length;
						for (int i = 1; i < len-1; i++) {
							double v = Double.parseDouble(tokens[i]);
							X[c][i-1] = v;
						}

						dens[c][0] = Double.parseDouble(tokens[len-1]);
						c++;
					}
					j++;
				}
				//System.out.println(X[0][0] + "  " + X[0][1] + "  " + dens[0][0]); 

			} finally {
				input.close();
			}
		} catch (IOException ex) {
			ex.printStackTrace();
		}
	}

	public void getData1() {

		// assumes: id,x1,x2,...,density  format, so dim = col - 2
		try {

			BufferedReader input = new BufferedReader(new FileReader(indist));
			try {
				String line = null;
				String[] tokens = null;

				int j = 0;
				while ((line = input.readLine()) != null) {
					j++;
				}
				m = j;			
				System.out.println("total num pts = " + m);
			} finally {
				input.close();
			}
		} catch (IOException ex) {
			ex.printStackTrace();
		}

		dist = new double[m][m];
		dens = new double[m][2];  

		// now read data;

		try {

			Random rand = new Random(0);
			double eps = Math.pow(10,-6);
			BufferedReader input = new BufferedReader(new FileReader(indist));
			try {
				String line = null;
				String[] tokens = null;

				int j = 0;
				while ((line = input.readLine()) != null) {
					tokens = line.split(inseparator);
					int len = tokens.length;
					for (int i = j; i < len; i++) {
						double v = Double.parseDouble(tokens[i]);
						dist[j][i] = v;  // + eps * rand.nextDouble();
						dist[i][j] = dist[j][i];
					}
					dens[j][0] = 0;
					j++;
				}
				//System.out.println(X[0][0] + "  " + X[0][1] + "  " + dens[0][0]); 

			} finally {
				input.close();
			}
		} catch (IOException ex) {
			ex.printStackTrace();
		}
	}

	public void printData() {		

		try {

			FileWriter outFile = new FileWriter(out);
			PrintWriter out = new PrintWriter(outFile);

			for (int i = 0; i < m; i++) {

				for (int j = 0; j < dim; j++) {
					out.print(X[i][j] + outseparator);
				}
				out.print(dens[i][0] + outseparator);
				out.println(dens[i][1]);
			}
			out.close();
		} catch (IOException ex) {
			ex.printStackTrace();
		}

	}

	public void printData1() {		

		try {

			FileWriter outFile = new FileWriter(outdist);
			PrintWriter out = new PrintWriter(outFile);

			dim = X[0].length;
			for (int i = 0; i < m; i++) {

				for (int j = 0; j < dim; j++) {
					out.print(X[i][j] + outseparator);
				}
				out.print(dens[i][0] + outseparator);
				out.println(dens[i][1]);
			}
			out.close();
		} catch (IOException ex) {
			ex.printStackTrace();
		}

	}

	public void computeDist() {

		for(int i=0;i<m-1;i++) {
			for(int j=i+1;j<m;j++) {
				dist[i][j] = L2_dist(X[i], X[j]);
				dist[j][i] = dist[i][j];
				diamX = (diamX >= dist[i][j]) ? diamX : dist[i][j];
			}
		}
	}

	public void meanNormalize() {

		double[] center = new double[dim];
		for(int i=0;i<m;i++) {
			for(int j=0;j<dim;j++) {
				center[j] += X[i][j];
			}
		}
		for(int j=0;j<dim;j++) {
			center[j] /= m;
		}
		for(int i=0;i<m;i++) {
			for(int j=0;j<dim;j++) {	
				X[i][j] -= center[j];
			}
		}
	}

	public int computeWeight(double[] x, double a, double[] v, double ff) {

		double[] w = mult(v, a);
		double[] diff = subtract(x, w);	
		double norm = Math.sqrt(dot(diff, diff));	
		double ratio = norm / diamX;
		int multiplicity = (int) Math.ceil(ff * (1 - ratio));

		return Math.max(multiplicity, 1);
	}

	public void process() {
		getData();
		computeDist();
		X = mds(dist);
		meanNormalize();
		computeVectorSpaceDensity0();
		printData();
	}

	public void process1() {
		getData1();
		X = mds(dist);
		meanNormalize();
		computeVectorSpaceDensity0();
		printData1();
	}

	public void computeVectorSpaceDensity0() {

		double eps = Math.pow(10, -6);

		double[][] v = new double[][]{ {1.0, 0}, {0, 1.0} };
		double ff = 10;

		int len = v.length;
		double[][] values = new double[m][len];

		for(int g=0;g<len;g++) {

			int T = 0;
			double[] data = new double[m];
			int[] multiplicity = new int[m];
			double h = 0;
			double diam = 0;

			for(int i=0;i<m;i++) {
				data[i] = dot(X[i], v[g]);
				multiplicity[i] = computeWeight(X[i], data[i], v[g], ff);
				T += multiplicity[i];
				System.out.println(i + "  :  " + multiplicity[i]);

			}

			double minx = Double.POSITIVE_INFINITY;
			double maxx = Double.NEGATIVE_INFINITY;
			double[] data1 = new double[T];

			for(int i=0;i<m;i++) {
				data1[i] = data[i];
				if(data1[i] > maxx) 
					maxx = data1[i];
				if(data1[i] < minx) 
					minx = data1[i];
			}
			int q = m;
			Random rand = new Random(0);
			for(int i=0;i<m;i++) {
				for(int j=0;j<multiplicity[i]-1;j++) {
					data1[q] = data[i] + eps * rand.nextDouble();
					if(data1[q] > maxx) 
						maxx = data1[q];
					if(data1[q] < minx) 
						minx = data1[q];
					q++;
				}
			}

			diam = maxx - minx;
			for(int i=0;i<T;i++) {
				data1[i] = (data1[i] - minx)/diam;
			}

			h = findBandwidth(data1, 0, 1);
			System.out.println("h ---> " + nf.format(h));

			for(int i=0;i<m;i++) {
				double value = 0.0;
				for(int k=0;k<m;k++) {    
					if(i != k) {
						double x = Math.abs(data1[i] - data1[k]);
						value += Math.exp(-x*x/(2*h));  
					}
				}
				value /= (Math.sqrt(2*Math.PI*h)*(m-1));
				values[i][g] = value;
			}
		}

		for(int i=0;i<m;i++) {
			for(int g=0;g<len;g++) {
				dens[i][1] += values[i][g];
			}
		}
		for(int i=0;i<m;i++) {
			dens[i][1] /= len;
			if(dens[i][1] < -1111.48) {
				dens[i][1] = 0;
			}
			//System.out.println(dens[i][1]);
		}

	}

	public void computeVectorSpaceDensity1() {

		double[] data = new double[m];
		double[] h = new double[dim];
		double[] diam = new double[dim];

		for(int j=0;j<dim;j++) {
			double minx = Double.POSITIVE_INFINITY;
			double maxx = Double.NEGATIVE_INFINITY;
			for(int i=0;i<m;i++) {
				data[i] = X[i][j];
				if(data[i] > maxx) 
					maxx = data[i];
				if(data[i] < minx) 
					minx = data[i];
			}
			for(int i=0;i<m;i++) {
				data[i] = (data[i] - minx)/(maxx - minx);
			}
			diam[j] = maxx - minx;

			h[j] = findBandwidth(data, 0, 1);
			System.out.println(j + " ---> " + nf.format(h[j]));
		}

		for(int i=0;i<m;i++) {
			double[] values = new double[dim];
			double value = 0.0;

			for(int k=0;k<m;k++) {    
				if(i != k) {
					for(int j=0;j<dim;j++) {
						double x = Math.abs(X[i][j] - X[k][j])/diam[j];
						values[j] += Math.exp(-x*x/(2*h[j]));  
					}
				}
			}
			for(int q=0;q<dim;q++) {
				values[q] /= (Math.sqrt(2*Math.PI*h[q])*(m-1));
				if(values[q] > 0)
					value += Math.log(values[q]);  // product of gaussian kernels in each dimension; to avoid small numbers a*b is replaced with log(a) + log(b) 
			}
			dens[i][1] = value; //Math.exp(value);  //  no need to apply exp() since it's monotone; only order is important here
			//System.out.println(dens[i][1]);
		}
	}

	/// mds functions
	//***************

	public double[][] mds(double[][] dist) {

		int mindim = 2;
		int maxdim = 2;

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
		int xdim = 0;
		int len = pair.diag.length;
		double maxEigenvalue = pair.diag[0];
		System.out.println("0 : " + maxEigenvalue);
		int T = Math.min(maxdim, len);
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
		xdim = Math.max(xdim, mindim);
		xdim = 2;
		System.out.println("embedding dim = " + xdim);

		double[][] M = new double[n][xdim];

		for(int i=0;i<n;i++) {
			for(int j=0;j<xdim;j++) {
				M[i][j] = Math.sqrt(pair.diag[j]) * pair.U[i][j];  
			}
		}

		return M;
	}

	////
	// density helper functions
	////*****************

	double findBandwidth(double[] data, double minx, double maxx) {

		int n = (int) Math.pow(2, 5);
		int N = data.length;
		double range = maxx - minx;
		double min = minx - range/10;
		double max = maxx + range/10;
		double r = max - min;
		double dx = r/(n-1);
		double[] mesh = new double[n];
		for(int q=0;q<n;q++)
			mesh[q] = min + q*dx;
		double[] idata = hist(data, mesh, 0);
		double isum = 0.0;
		for(int i=0;i<idata.length;i++)
			isum += idata[i];
		idata = mult(idata, 1/isum);

		double[] I = new double[n-1];
		for(int i=0;i<n-1;i++) {
			I[i] = (i+1)*(i+1);
		}
		double[] a = dct(idata);
		double[] a2 = new double[a.length-1];
		//		for(int ii=0;ii<a.length;ii++)  {
		//			System.out.println(ii + " --> " + nf.format(a[ii]));
		//		}
		for(int i=0;i<a2.length;i++)
			a2[i] = (a[i+1]/2.0) * (a[i+1]/2.0);

		double h = solve(N, I, a2);		
		//System.out.println("h0 = " + nf.format(h));

		h = Math.sqrt(h)*r;

		return h;
	}

	double[] hist(double[] data, double[] mesh, int mode) {

		int dlen = data.length;
		int len = mesh.length;
		double[] h = new double[len];
		for(int i=0;i<dlen;i++) {
			int q = 1;
			while(data[i] > mesh[q])
				q++;
			h[q-1] += 1;
		}

		if(mode == 0) {
			for(int i=0;i<len;i++)
				h[i] /= dlen;
		}

		return h;
	}

	double[] dct(double[] data) {

		int len = data.length;
		ArrayList<Double> dd = new ArrayList<Double>();
		Complex[] w = new Complex[len];
		w[0] = new Complex(1.0, 0);
		for(int i=1;i<len;i++) {
			Complex v = Complex.I.multiply(-i*Math.PI/(2*len));
			w[i] = v.exp().multiply(2);
		}

		int j = 0;
		while(j < len) {
			dd.add(data[j]);
			j += 2;
		}
		j = len - 1;
		while(j >= 1) {
			dd.add(data[j]);
			j -= 2;
		}
		double[] data1 = new double[dd.size()];
		for(int i=0;i<dd.size();i++)
			data1[i] = dd.get(i);

		//		for(double d1 : data)
		//			System.out.println(nf.format(d1));
		//		System.out.println("*************" + len + " " + dd.size());
		//		for(double d1 : dd)
		//			System.out.println(nf.format(d1));

		FastFourierTransformer fft = new FastFourierTransformer();
		Complex[] z = fft.transform(data1);

		double[] real = new double[z.length];
		for(int i=0;i<z.length;i++) {			
			real[i] = w[i].getReal()*z[i].getReal() - w[i].getImaginary()*z[i].getImaginary();
		}
		return real;
	}

	double solve(int N, double[] I, double[] a2) {

		//		System.out.println(N);
		//		for(double g=0;g<=1;g+=0.05) {
		//			double fv = f(g, N, I, a2);
		//			System.out.println(nf.format(g) + " " + nf.format(fv));
		//		}
		//		for(int ii=0;ii<I.length;ii++)  {
		//			System.out.println(I[ii]);
		//		}
		//		for(int ii=0;ii<a2.length;ii++)  {
		//			System.out.println(ii + " --> " + nf.format(a2[ii]));
		//		}

		double eps = Math.pow(10, -4);
		double a = Math.pow(10, -4);
		double b = 0.2;
		double q = -1;
		int iternum = 20;
		int k = 0;
		double va = f(a, N, I, a2);
		double vb = f(b, N, I, a2);
		double vq = 0;
		while(k < iternum && Math.abs(va) > eps && Math.abs(vb) > eps) {

			q = (va + vb)/2.0;
			vq = f(q, N, I, a2);
			if(vq*va > 0) {
				a = q;
				va = f(a, N, I, a2);
			}
			else {
				b = q;
				vb = f(b, N, I, a2);
			}
			k++;
			//System.out.println("va = " + nf.format(va));

		}
		if(q >= 0) return q;
		else if(Math.abs(va) < Math.abs(vb)) return a;
		else return b;
	}

	double f(double t, int N, double[] I, double[] a2) {
		int p = 7;
		int len = I.length;
		double sum = 0;
		for(int i=0;i<len;i++) {
			double w = -I[i]*Math.PI*Math.PI*t;
			double z = Math.exp(w)*a2[i]*Math.pow(I[i], p);
			sum += z;
		}
		double f = 2*sum*Math.pow(Math.PI, 2*p);

		for(int s=p-1;s>=2;s--) {
			double K = 1.0;
			for(int i=1;i<=2*s-1;i+=2)
				K *= i;
			K /= Math.sqrt(2*Math.PI);
			double cst = (1 + Math.pow(0.5, s+0.5))/3.0;
			double time = Math.pow(2*cst*K/(N*f), 2.0/(3.0+2.0*s));
			sum = 0;
			for(int i=0;i<len;i++) {
				double w = -I[i]*Math.PI*Math.PI*time;
				double z = Math.exp(w)*a2[i]*Math.pow(I[i], s);
				sum += z;
			}
			f = 2*sum*Math.pow(Math.PI, 2.0*s);
		}

		double v = t - Math.pow(2*N*Math.sqrt(Math.PI)*f, (-2.0/5));
		return v;
	}

	double[] mult(double[] x, double a) {

		int len = x.length;
		double[] y = new double[len];
		for(int i=0;i<len;i++)
			y[i] = a * x[i];

		return y;
	}

	public Pair svd(double[][] x) {

		int len = x.length;
		int dim = x[0].length;
		//		double[][] x0 = new double[len][dim]; // centered
		//		double[] means = new double[dim];
		//		for(int i=0;i<dim;i++) {
		//			for(int j=0;j<len;j++) {
		//				means[i] += x[j][i];
		//			}
		//		}
		//
		//		for(int j=0;j<len;j++) {
		//			for(int i=0;i<dim;i++) {
		//				x0[j][i] = x[j][i] - means[i]/len;
		//			}
		//		}

		Matrix AM = new Matrix(x);
		//Matrix covA = AM.transpose().times(AM);
		SingularValueDecomposition svd = new SingularValueDecomposition(AM);
		Matrix U = svd.getU();
		double[][] u = U.getArray();
		double[] diag = svd.getSingularValues();

		return new Pair(u, diag);

		//		double[] svd1 = new double[U2.getRowDimension()];
		//		for(int i=0;i<U2.getRowDimension();i++) {
		//			svd1[i] = U2.get(i, 0);  // 1st svd vector
		//		}
		//
		//		return svd1;	
	}

	public int[] shuffle(int len, int seed) {
		Random rand = new Random(seed);
		int[] ar = new int[len];
		for (int i = 0; i < len; i++)
			ar[i] = i;
		for (int i = len - 1; i >= 1; i--) {
			int j = rand.nextInt(i + 1);
			int t = ar[i];
			ar[i] = ar[j];
			ar[j] = t;
		}
		return ar;
	}

	public double dot(double[] v, double[] w) {
		double value = 0;
		for(int i=0;i<v.length;i++) {
			value += v[i]*w[i];
		}
		return value;
	}

	public double[] subtract(double[] v, double[] w) {
		double[] diff = new double[v.length];
		for(int i=0;i<v.length;i++) {
			diff[i] = v[i] - w[i];
		}
		return diff;
	}


	//	double L1_dist(double[] x, double[] y) {
	//		double dist = 0;
	//		for(int i=0;i<x.length;i++)
	//			dist += Math.abs((x[i] - y[i]));
	//		return dist/(double)x.length;
	//	}
	//
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

	public static void main(String[] args) {

		double scale = 1.0;
		Density dd = new Density(scale);
		//dd.process();
		dd.process1();
	}

}


