import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import 'bootstrap/dist/css/bootstrap.min.css';

// Utility: Calculate how many days the current stock will last
const calculateDaysToReplenish = (stock, avgWeeklySales) => {
    const safeStock = stock || 0;
    const safeSales = avgWeeklySales || 0;

    if (safeSales === 0) return 999; // If no sales, stock lasts forever

    const dailySales = safeSales / 7;
    return Math.floor(safeStock / dailySales); // Round down to nearest whole day
};

export default function InventoryPredictor() {
    const [products, setProducts] = useState([]);
    const [predictionResults, setPredictionResults] = useState([]);
    const [loading, setLoading] = useState(true);
    const [predicting, setPredicting] = useState(false);
    const [error, setError] = useState(null);

    // Fetch product data from API
    useEffect(() => {
        fetch('https://my.api.mockaroo.com/product_sample?key=c29c3fd0')
            .then(res => {
                if (!res.ok) {
                    throw new Error(`Fetch Failed!: ${res.status} ${res.statusText}`);
                }
                return res.json();
            })
            .then(data => {
                // Calculate "Days to Replenish" immediately upon fetching
                const processedData = data.map(p => ({
                    ...p,
                    days_to_replenish: calculateDaysToReplenish(p.stock, p.average_weekly_sales)
                }));
                setProducts(processedData);
                setLoading(false);
            })
            .catch((err) => {
                setError(err.message || 'Error fetching products');
                setLoading(false);
            });
    }, []);

    // Predict reorder logic
    const handlePredict = async () => {
        if (products.length === 0) return;
        setPredicting(true);

        // Prepare training data
        const trainingInputs = products.map(p => [
            p.stock,
            p.average_weekly_sales || 0,
            p.lead_time || 0,
        ]);

        // **Updated Logic:** // We train the AI to trigger a reorder if "Days to Replenish" is less than the "Lead Time"
        const trainingOutputs = products.map(p => {
            const daysLeft = calculateDaysToReplenish(p.stock, p.average_weekly_sales);
            // If we have fewer days of stock left than it takes to ship new stock, REORDER (1)
            return daysLeft <= (p.lead_time || 0) ? 1 : 0;
        });

        // Convert to tensors
        const trainXs = tf.tensor2d(trainingInputs);
        const trainYs = tf.tensor2d(trainingOutputs, [trainingOutputs.length, 1]);

        // Build & train model
        const model = tf.sequential();
        model.add(tf.layers.dense({ inputShape: [3], units: 8, activation: 'relu' }));
        model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
        model.compile({
            optimizer: 'adam',
            loss: 'binaryCrossentropy',
            metrics: ['accuracy'],
        });

        await model.fit(trainXs, trainYs, { epochs: 50, shuffle: true });

        trainXs.dispose();
        trainYs.dispose();

        // Predict results
        const preds = [];
        for (const p of products) {
            const inputTensor = tf.tensor2d([[p.stock, p.average_weekly_sales || 0, p.lead_time || 0]]);
            const predVal = (await model.predict(inputTensor).data())[0];
            inputTensor.dispose();

            const daysLeft = calculateDaysToReplenish(p.stock, p.average_weekly_sales);

            preds.push({
                name: p.name,
                prediction: predVal > 0.5 ? 'Reorder' : 'No Reorder',
                stock: p.stock,
                avgSales: p.average_weekly_sales,
                leadTime: p.lead_time,
                days_to_replenish: daysLeft, 
            });
        }
        setPredictionResults(preds);
        setPredicting(false);
    };

    if (loading) {
        return (
            <div className="d-flex justify-content-center align-items-center" style={{ minHeight: '60vh' }}>
                <div className="spinner-border text-primary" role="status" style={{ width: 70, height: 70 }}>
                    <span className="visually-hidden">Loading...</span>
                </div>
            </div>
        );
    }

    if (error) {
        return <div className="alert alert-danger text-center">{error}</div>;
    }

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
                        <span className="spinner-border spinner-border-sm me-2" />
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
                            {/* Updated Header */}
                            <th>Days to Replenish</th> 
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
                                    {/* Updated Display Logic */}
                                    <span className={
                                        // Highlight RED if days left is less than lead time
                                        p.days_to_replenish <= (p.leadTime || p.lead_time) 
                                        ? 'text-danger fw-bold' 
                                        : ''
                                    }>
                                        {p.days_to_replenish === 999 ? '> 999' : p.days_to_replenish} Days
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
