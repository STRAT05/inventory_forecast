import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import 'bootstrap/dist/css/bootstrap.min.css';

// Utility function to calculate Reorder Point (or 'days_to_replenish' column in mock data)
const calculateReorderPoint = (avgSales, leadTime) => {
    // Ensure both values are treated as numbers, defaulting to 0
    const sales = avgSales || 0;
    const time = leadTime || 0;
    // The Reorder Point is: Average Weekly Sales * Lead Time
    return (sales * time); 
};

export default function InventoryPredictor() {
    const [products, setProducts] = useState([]);
    const [predictionResults, setPredictionResults] = useState([]);
    const [loading, setLoading] = useState(true);
    const [predicting, setPredicting] = useState(false);
    const [error, setError] = useState(null);

    // Fetch product data from API
    useEffect(() => {
        fetch('https://my.api.mockaroo.com/product_sample?key=c29c3fd0') // fetch products from local API
            .then(res => { // if response not ok, throw error else return json
                if (!res.ok) {
                    throw new Error(`Fetch Failed!: ${res.status} ${res.statusText}`);
                }
                return res.json();
            })
            .then(data => {
                // Add the calculated Reorder Point (Days to Replenish) to the product data
                const productsWithReorderPoint = data.map(p => ({
                    ...p,
                    reorder_point: calculateReorderPoint(p.average_weekly_sales, p.lead_time)
                }));
                setProducts(productsWithReorderPoint);
                setLoading(false);
            })
            .catch((err) => {
                setError(err.message || 'Error fetching products');
                setLoading(false);
            });
    }, []);

    // Predict reorder for all products with clean TensorFlow handling
    const handlePredict = async () => {
        if (products.length === 0) return;
        setPredicting(true);

        // Prepare training data
        const trainingInputs = products.map(p => [
            p.stock,
            p.average_weekly_sales || 0,
            p.lead_time || 0,
        ]);
        
        // **Updated Training Output (using Reorder Point formula)**
        const trainingOutputs = products.map(p => {
            const reorderPoint = calculateReorderPoint(p.average_weekly_sales, p.lead_time);
            return p.stock <= reorderPoint ? 1 : 0; // 1 (Reorder) if stock <= Reorder Point
        });

        // Convert to tensors
        const trainXs = tf.tensor2d(trainingInputs);
        const trainYs = tf.tensor2d(trainingOutputs, [trainingOutputs.length, 1]);

        // Build & train model (epochs lowered for faster response)
        const model = tf.sequential();
        model.add(tf.layers.dense({ inputShape: [3], units: 8, activation: 'relu' }));
        model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
        model.compile({
            optimizer: 'adam',
            loss: 'binaryCrossentropy',
            metrics: ['accuracy'],
        });

        await model.fit(trainXs, trainYs, { epochs: 80, shuffle: true }); // reduced epochs for speed

        // Dispose training tensors to free memory
        trainXs.dispose();
        trainYs.dispose();

        // Predict results (dispose tensors for memory safety)
        const preds = [];
        for (const p of products) {
            const inputTensor = tf.tensor2d([[p.stock, p.average_weekly_sales || 0, p.lead_time || 0]]);
            const predVal = (await model.predict(inputTensor).data())[0];
            inputTensor.dispose();

            // Calculate Reorder Point for display
            const reorderPoint = calculateReorderPoint(p.average_weekly_sales, p.lead_time);

            preds.push({
                name: p.name,
                prediction: predVal > 0.5 ? 'Reorder' : 'No Reorder',
                stock: p.stock,
                avgSales: p.average_weekly_sales,
                leadTime: p.lead_time,
                reorder_point: reorderPoint, // Include Reorder Point in prediction results
                
            });
        }
        setPredictionResults(preds);
        setPredicting(false);
    };

    // Loading UI
    if (loading) {
        return (
            <div className="d-flex justify-content-center align-items-center" style={{ minHeight: '60vh' }}>
                <div className="spinner-border text-primary" role="status" style={{ width: 70, height: 70 }}>
                    <span className="visually-hidden">Loading...</span>
                </div>
            </div>
        );
    }

    // Error UI
    if (error) {
        return (
            <div className="alert alert-danger text-center" role="alert">
                {error}
            </div>
        );
    }

    // Component UI
    return (
        <div className="container py-4">
            <h2 className="mb-4 text-center">Inventory Reorder Predictor</h2>
            <button
                className="btn btn-primary mb-3"
                onClick={handlePredict}
                disabled={predicting}
            >
                {predicting ? (
                    <>
                        <span className="spinner-border spinner-border-sm me-2" role="status" />
                        Predicting...
                    </>
                ) : 'Predict'}
            </button>

            <div className="table-responsive">
                <table className="table table-bordered table-hover">
                    <thead className="table-light">
                        <tr>
                            <th>Name</th>
                            <th>Stock</th>
                            <th>Avg Weekly Sales</th>
                            <th>Lead Time (days)</th>
                            <th>Days to Replenish (Units)</th> {/* NEW COLUMN */}
                            <th>Prediction</th>
                        </tr>
                    </thead>
                    <tbody>
                        {(predictionResults.length > 0 ? predictionResults : products).map((p, idx) => (
                            <tr key={p.name || idx}>
                                <td>{p.name}</td>
                                <td>{p.stock}</td>
                                <td>{p.avgSales || p.average_weekly_sales}</td>
                                <td>{p.leadTime || p.lead_time}</td>
                                <td>
                                    {/* Display the calculated Reorder Point */}
                                    <span className={p.stock <= p.reorder_point ? 'text-danger fw-bold' : ''}>
                                        {p.reorder_point}
                                    </span>
                                </td> 
                                <td>
                                    {p.prediction ? (
                                        <span className={p.prediction === 'Reorder' ? 'text-danger fw-bold' : 'text-success fw-bold'}>
                                            {p.prediction}
                                        </span>
                                    ) : (
                                        '-'
                                    )}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

