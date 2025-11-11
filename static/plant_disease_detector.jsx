/**
 * Plant Disease Detector - Full-Stack React Component
 *
 * This React app runs a local ONNX model in the browser for fast detection
 * and then calls a separate Flask backend API to securely get a
 * Gemini-powered treatment plan.
 */

// --- CONFIGURATION ---------------------------------------------------

// 1. CLASS NAMES
const CLASS_NAMES = [
  'Apple___Apple_scab',
  'Apple___Black_rot',
  'Apple___Cedar_apple_rust',
  'Apple___healthy',
  'Blueberry___healthy',
  'Cherry_(including_sour)___Powdery_mildew',
  'Cherry_(including_sour)___healthy',
  'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
  'Corn_(maize)___Common_rust_',
  'Corn_(maize)___Northern_Leaf_Blight',
  'Corn_(maize)___healthy',
  'Grape___Black_rot',
  'Grape___Esca_(Black_Measles)',
  'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
  'Grape___healthy',
  'Orange___Haunglongbing_(Citrus_greening)',
  'Peach___Bacterial_spot',
  'Peach___healthy',
  'Pepper,_bell___Bacterial_spot',
  'Pepper,_bell___healthy',
  'Potato___Early_blight',
  'Potato___Late_blight',
  'Potato___healthy',
  'Raspberry___healthy',
  'Soybean___healthy',
  'Squash___Powdery_mildew',
  'Strawberry___Leaf_scorch',
  'Strawberry___healthy',
  'Tomato___Bacterial_spot',
  'Tomato___Early_blight',
  'Tomato___Late_blight',
  'Tomato___Leaf_Mold',
  'Tomato___Septoria_leaf_spot',
  'Tomato___Spider_mites Two-spotted_spider_mite',
  'Tomato___Target_Spot',
  'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
  'Tomato___Tomato_mosaic_virus',
  'Tomato___healthy'
];

// 2. MODEL PATHS
const MODEL_PATH = './model.onnx';
const MODEL_DATA_PATH = './model.onnx.data'; // The external weights file

// --- UPDATED: Use a relative path for the API ---
const FLASK_API_URL = '/api/get_treatment';

// 4. MODEL INPUT/OUTPUT
const MODEL_WIDTH = 224;
const MODEL_HEIGHT = 224;
const mean = [0.485, 0.456, 0.406];
const std = [0.229, 0.224, 0.225];

// --- HELPER COMPONENTS ----------------------------------------------

/**
 * A simple, reusable spinning loader component.
 */
function Loader() {
  return (
    <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" role="status">
      <span className="sr-only">Loading...</span>
    </div>
  );
}

/**
 * A reusable error banner.
 * @param {{ message: string }} props
 */
function ErrorBanner({ message }) {
  if (!message) return null;
  return (
    <div className="p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg" role="alert">
      <p className="font-bold">Error:</p>
      <p>{message}</p>
    </div>
  );
}

/**
 * A component to display the treatment plan, including sources.
 * @param {{ plan: string, sources: Array<{uri: string, title: string}> }} props
 */
function TreatmentPlan({ plan, sources }) {
  if (!plan) return null;
  const formattedPlan = plan;

  return (
    <div className="mt-6 pt-6 border-t border-gray-300">
      <h3 className="text-2xl font-bold text-gray-800 mb-4">Recommended Treatment Plan</h3>
      <div 
        className="p-4 bg-blue-50 text-blue-900 rounded-lg space-y-2 whitespace-pre-wrap"
        dangerouslySetInnerHTML={{ __html: formattedPlan }}
      />
      
      {sources && sources.length > 0 && (
        <div className="mt-4">
          <h4 className="text-sm font-semibold text-gray-600">Sources (from Google Search):</h4>
          <ul className="list-disc list-inside text-sm mt-2">
            {sources.map((source, index) => (
              <li key={index} className="truncate">
                <a 
                  href={source.uri} 
                  target="_blank" 
                  rel="noopener noreferrer" 
                  className="text-blue-600 hover:underline"
                >
                  {source.title || source.uri}
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

/**
 * A component to display the prediction result.
 * @param {{ result: {disease: string, confidence: number} }} props
 */
function PredictionResult({ result }) {
  if (!result) return null;
  
  const isHealthy = result.disease.toLowerCase().includes('healthy');
  const confidencePct = (result.confidence * 100).toFixed(2);
  const colorClass = isHealthy ? 'text-green-600' : 'text-red-600';

  return (
    <div className="text-center">
      <h3 className="text-xl font-medium text-gray-700">Diagnosis:</h3>
      <p className={`text-3xl font-bold ${colorClass}`}>{result.disease}</p>
      
      <h3 className="text-xl font-medium text-gray-700 mt-4">Confidence:</h3>
      <p className={`text-3xl font-bold ${colorClass}`}>{confidencePct}%</p>
    </div>
  );
}


// --- MAIN APP COMPONENT ---------------------------------------------

function App() {
  const { useState, useEffect, useRef } = React;

  // --- State Variables ---
  // For the ONNX model
  const [session, setSession] = useState(null);
  const [modelError, setModelError] = useState(null);
  const [modelLoading, setModelLoading] = useState(true); // Start loading on mount

  // For the image and prediction
  const [imageFile, setImageFile] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [predictError, setPredictError] = useState(null);
  const [predictLoading, setPredictLoading] = useState(false);

  // For the Gemini treatment plan
  const [treatment, setTreatment] = useState(null);
  const [treatmentSources, setTreatmentSources] = useState([]);
  const [treatmentError, setTreatmentError] = useState(null);
  const [treatmentLoading, setTreatmentLoading] = useState(false);

  // Ref to the hidden canvas for image processing
  const canvasRef = useRef(null);

  // Load the ONNX model ---
  // This runs once when the component first mounts.
  useEffect(() => {
    (async () => {
      try {
        if (!window.ort) {
          throw new Error("ONNX Runtime (ort) is not loaded. Check your index.html.");
        }

        setModelLoading(true);
        setModelError(null);
                
        // Manually fetch both model files as raw byte buffers
        const [modelResponse, dataResponse] = await Promise.all([
          fetch(MODEL_PATH),
          fetch(MODEL_DATA_PATH)
        ]);

        if (!modelResponse.ok) throw new Error(`Failed to fetch model.onnx: ${modelResponse.status}`);
        if (!dataResponse.ok) throw new Error(`Failed to fetch model.onnx.data: ${dataResponse.status}`);

        const modelBuffer = await modelResponse.arrayBuffer();
        const dataBuffer = await dataResponse.arrayBuffer();
        
        // Set options for ONNX Runtime
        const options = {
          executionProviders: ['wasm'], // 'wasm' is the most compatible (CPU)
          graphOptimizationLevel: 'all',
          
          // Pass the .data buffer as 'externalData'
          externalData: [
            {
              data: dataBuffer,
              path: 'model.onnx.data', // The name the .onnx file expects
            },
          ],
        };

        // Create the session from the model *buffer*
        const inferenceSession = await ort.InferenceSession.create(modelBuffer, options);
        
        setSession(inferenceSession);
        setModelError(null);

      } catch (e) {
        console.error(e);
        let errorString = "Unknown error";
        if (e instanceof Error) {
          errorString = e.message;
          if (e.message.includes('404') || e.message.includes('fetch')) {
             errorString = "Model file not found (404). Make sure 'model.onnx' AND 'model.onnx.data' are in the same folder as index.html.";
          }
        }
        setModelError(errorString);
      } finally {
        setModelLoading(false);
      }
    })();
  }, []); // Empty array means this runs only once on mount


  // --- Event Handler: Image Upload ---
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      setImageUrl(URL.createObjectURL(file));
      // Clear all old results
      setPrediction(null);
      setPredictError(null);
      setTreatment(null);
      setTreatmentError(null);
      setTreatmentSources([]);
    }
  };

  // --- Core Logic 1: Pre-process Image ---
  /**
   * Converts the uploaded image file into a model-ready tensor.
   * @param {File} file - The image file from the input.
   * @returns {Promise<ort.Tensor>} A 1x3x224x224 tensor.
   */
  const preprocessImage = (file) => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.src = URL.createObjectURL(file);
      img.onload = () => {
        try {
          const canvas = canvasRef.current;
          if (!canvas) throw new Error("Canvas ref is not available");
          
          const ctx = canvas.getContext('2d', { willReadFrequently: true });
          canvas.width = MODEL_WIDTH;
          canvas.height = MODEL_HEIGHT;

          // Draw the image onto the canvas, resizing it
          ctx.drawImage(img, 0, 0, MODEL_WIDTH, MODEL_HEIGHT);
          
          // Get pixel data
          const imgData = ctx.getImageData(0, 0, MODEL_WIDTH, MODEL_HEIGHT);
          const pixels = imgData.data;

          // Pre-allocate the tensor data array
          const float32Data = new Float32Array(1 * 3 * MODEL_WIDTH * MODEL_HEIGHT);
          
          let dataIndex = 0;

          // Loop through pixels and apply normalization
          // This performs ToTensor() and Normalize() in one pass
          for (let c = 0; c < 3; c++) { // For each channel (R, G, B)
            for (let h = 0; h < MODEL_HEIGHT; h++) { // For each row
              for (let w = 0; w < MODEL_WIDTH; w++) { // For each column
                const pixelIndex = (h * MODEL_WIDTH + w) * 4; // *4 for R,G,B,A
                const pixelValue = pixels[pixelIndex + c] / 255; // Get R, G, or B, scale 0-1
                const normalizedValue = (pixelValue - mean[c]) / std[c];
                float32Data[dataIndex++] = normalizedValue;
              }
            }
          }
          
          // Create the ONNX Tensor
          const tensor = new ort.Tensor('float32', float32Data, [1, 3, MODEL_WIDTH, MODEL_HEIGHT]);
          resolve(tensor);

        } catch (e) {
          reject(e);
        }
      };
      img.onerror = (e) => {
        reject(new Error("Failed to load image for processing."));
      };
    });
  };

  // --- Core Logic 2: Post-process Model Output ---
  /**
   * Converts the model's raw output (logits) into a readable prediction.
   * @param {ort.Tensor} output - The output tensor from the model.
   * @returns {{disease: string, confidence: number}}
   */
  const postprocessOutput = (output) => {
    const logits = output.data;
    
    // 1. Softmax (convert logits to probabilities)
    // (This is a numerically stable softmax)
    const maxLogit = Math.max(...logits);
    const exps = logits.map(logit => Math.exp(logit - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b);
    const probabilities = exps.map(exp => exp / sumExps);

    // 2. Find the max probability
    let maxProb = 0;
    let maxIndex = 0;
    probabilities.forEach((prob, i) => {
      if (prob > maxProb) {
        maxProb = prob;
        maxIndex = i;
      }
    });

    // 3. Map to class name
    return {
      disease: CLASS_NAMES[maxIndex] || 'Unknown',
      confidence: maxProb,
    };
  };

  // --- Core Logic 3: Run Prediction (ONNX) ---
  const runPrediction = async () => {
    if (!imageFile || !session) return;

    setPredictLoading(true);
    setPredictError(null);
    setPrediction(null);
    setTreatment(null);
    setTreatmentError(null);
    setTreatmentSources([]);

    try {
      // 1. Pre-process the image
      const tensor = await preprocessImage(imageFile);

      // 2. Prepare model inputs
      const feeds = { [session.inputNames[0]]: tensor };

      // 3. Run the model
      const results = await session.run(feeds);

      // 4. Get the output
      const output = results[session.outputNames[0]];

      // 5. Post-process and set state
      const result = postprocessOutput(output);
      setPrediction(result);

    } catch (e) {
      console.error("Prediction error:", e);
      setPredictError(e.message || "An unknown error occurred during prediction.");
    } finally {
      setPredictLoading(false);
    }
  };

  // --- Core Logic 4: Get Treatment Plan (Gemini) ---
  const getTreatmentPlan = async () => {
    if (!prediction || prediction.disease.toLowerCase().includes('healthy')) {
      return;
    }
    
    setTreatmentLoading(true);
    setTreatmentError(null);
    setTreatment(null);
    setTreatmentSources([]);

    try {
      // --- This is the NEW part ---
      // Call our Flask backend API instead of Gemini directly
      const response = await fetch(FLASK_API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ disease: prediction.disease }),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || `API Error: ${response.status}`);
      }

      const data = await response.json();
      setTreatment(data.plan);
      setTreatmentSources(data.sources || []);

    } catch (e) {
      console.error("Treatment plan error:", e);
      // This is the error message the user sees
      setTreatmentError(e.message || "Failed to get treatment plan. Check connection or Flask server.");
    } finally {
      setTreatmentLoading(false);
    }
  };
  
  // --- JSX: Render the UI ---
  
  // Show a top-level error if the model fails to load
  if (modelError) {
    return (
      <div className="max-w-lg mx-auto my-10 p-8 bg-white rounded-lg shadow-xl">
        <ErrorBanner message={modelError} />
      </div>
    );
  }

  // Show a top-level loader while model is loading
  if (modelLoading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen">
        <Loader />
        <p className="text-gray-600 mt-4">Loading Disease Detection Model...</p>
      </div>
    );
  }

  // Model is loaded, show the main app
  return (
    <div className="max-w-2xl mx-auto my-10 p-8 bg-white rounded-lg shadow-xl">
      <canvas ref={canvasRef} className="hidden"></canvas>
      
      <h1 className="text-3xl font-bold text-green-700 text-center mb-6">
        Plant Disease Detector
      </h1>

      {/* --- Image Upload --- */}
      <div className="space-y-4">
        <label htmlFor="file-upload" className="block text-sm font-medium text-gray-700">
          Upload a picture of a plant leaf:
        </label>
        <input 
          id="file-upload" 
          type="file" 
          accept="image/*" 
          onChange={handleImageUpload}
          className="block w-full text-sm text-gray-500
                     file:mr-4 file:py-2 file:px-4
                     file:rounded-lg file:border-0
                     file:text-sm file:font-semibold
                     file:bg-green-50 file:text-green-700
                     hover:file:bg-green-100"
        />
        
        {imageUrl && (
          <div className="mt-4 p-4 border border-gray-200 rounded-lg">
            <img src={imageUrl} alt="Uploaded leaf" className="max-h-64 w-auto mx-auto rounded-md shadow-md" />
          </div>
        )}

        <button 
          onClick={runPrediction} 
          disabled={!imageFile || predictLoading}
          className="w-full flex justify-center py-3 px-4 border border-transparent
                     rounded-lg shadow-sm text-lg font-medium text-white bg-blue-600
                     hover:bg-blue-700 focus:outline-none focus:ring-2
                     focus:ring-offset-2 focus:ring-blue-500
                     disabled:bg-gray-400 disabled:cursor-not-allowed"
        >
          {predictLoading ? 'Analyzing...' : 'Detect Disease'}
        </button>
      </div>

      {/* --- Prediction Results Area --- */}
      <div className="mt-8 pt-6 border-t border-gray-300">
        {predictLoading && (
          <div className="flex justify-center">
            <Loader />
          </div>
        )}
        <ErrorBanner message={predictError} />
        <PredictionResult result={prediction} />
        
        {/* --- Treatment Plan Button --- */}
        {prediction && !prediction.disease.toLowerCase().includes('healthy') && (
          <button
            onClick={getTreatmentPlan}
            disabled={treatmentLoading}
            className="w-full flex justify-center py-3 px-4 border border-transparent
                       rounded-lg shadow-sm text-lg font-medium text-white bg-green-600
                       hover:bg-green-700 focus:outline-none focus:ring-2
                       focus:ring-offset-2 focus:ring-green-500 mt-6
                       disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            {treatmentLoading ? 'Getting Plan...' : '✨ Get Treatment Plan'}
          </button>
        )}
      </div>

      {/* --- Treatment Plan Results Area --- */}
      {treatmentLoading && (
        <div className="flex justify-center mt-6">
          <Loader />
        </div>
      )}
      <ErrorBanner message={treatmentError} />
      <TreatmentPlan plan={treatment} sources={treatmentSources} />
    </div>
  );
}

// --- Mount the App ---
// This is the final step that tells React to render the <App /> component
// into the <div id="root"></div> in your index.html.
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);