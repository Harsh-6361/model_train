# API Documentation

## Overview

The Multi-Modal ML Training System provides a REST API for model inference. The API is built with FastAPI and supports both tabular and image predictions.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. In production, consider adding:
- API keys
- OAuth 2.0
- JWT tokens

## Endpoints

### Root

#### `GET /`

Get API information and available endpoints.

**Response:**
```json
{
  "message": "Model Training API",
  "version": "0.1.0",
  "endpoints": {
    "health": "/health",
    "predict_tabular": "/predict/tabular (POST)",
    "predict_image": "/predict/image (POST)",
    "metrics": "/metrics"
  }
}
```

---

### Health Check

#### `GET /health`

Check the health status of the service and model.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.123456",
  "model_loaded": true
}
```

**Status Codes:**
- `200 OK`: Service is healthy
- `503 Service Unavailable`: Service is unhealthy

**Health Status Values:**
- `healthy`: All systems operational
- `warning`: Some issues detected (high resource usage)
- `error`: Critical issues (model not loaded, etc.)

---

### Metrics

#### `GET /metrics`

Get service metrics in Prometheus-compatible format.

**Response:**
```json
{
  "metrics": {
    "predict_tabular_latency": {
      "mean": 0.0123,
      "std": 0.0034,
      "min": 0.0089,
      "max": 0.0201,
      "last": 0.0115,
      "count": 100
    }
  },
  "counters": {
    "predict_tabular_requests": 100,
    "predict_image_requests": 50,
    "predict_tabular_errors": 2
  }
}
```

---

### Tabular Prediction

#### `POST /predict/tabular`

Make predictions on tabular data.

**Request Body:**
```json
{
  "features": [
    [5.1, 3.5, 1.4, 0.2],
    [6.3, 2.9, 5.6, 1.8]
  ]
}
```

**Parameters:**
- `features` (array of arrays): Feature vectors for prediction. Each inner array represents one sample.

**Response:**
```json
{
  "predictions": [
    {
      "predicted_class": 0,
      "predicted_label": "setosa",
      "confidence": 0.95,
      "probabilities": [0.95, 0.03, 0.02],
      "top_3": [
        {"class": 0, "label": "setosa", "probability": 0.95},
        {"class": 1, "label": "versicolor", "probability": 0.03},
        {"class": 2, "label": "virginica", "probability": 0.02}
      ]
    },
    {
      "predicted_class": 2,
      "predicted_label": "virginica",
      "confidence": 0.88,
      "probabilities": [0.02, 0.10, 0.88],
      "top_3": [
        {"class": 2, "label": "virginica", "probability": 0.88},
        {"class": 1, "label": "versicolor", "probability": 0.10},
        {"class": 0, "label": "setosa", "probability": 0.02}
      ]
    }
  ],
  "model_type": "tabular",
  "inference_time_ms": 12.5
}
```

**Status Codes:**
- `200 OK`: Prediction successful
- `400 Bad Request`: Invalid input format
- `500 Internal Server Error`: Prediction failed
- `503 Service Unavailable`: Model not loaded

**Example using cURL:**
```bash
curl -X POST "http://localhost:8000/predict/tabular" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[5.1, 3.5, 1.4, 0.2]]
  }'
```

**Example using Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict/tabular",
    json={"features": [[5.1, 3.5, 1.4, 0.2]]}
)
print(response.json())
```

---

### Image Prediction

#### `POST /predict/image`

Make predictions on image data.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Image file

**Response:**
```json
{
  "predictions": [
    {
      "predicted_class": 1,
      "predicted_label": "cat",
      "confidence": 0.92,
      "probabilities": [0.05, 0.92, 0.03],
      "top_3": [
        {"class": 1, "label": "cat", "probability": 0.92},
        {"class": 0, "label": "dog", "probability": 0.05},
        {"class": 2, "label": "bird", "probability": 0.03}
      ]
    }
  ],
  "model_type": "vision",
  "inference_time_ms": 45.2
}
```

**Status Codes:**
- `200 OK`: Prediction successful
- `400 Bad Request`: Invalid image format
- `500 Internal Server Error`: Prediction failed
- `503 Service Unavailable`: Model not loaded

**Supported Image Formats:**
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)

**Example using cURL:**
```bash
curl -X POST "http://localhost:8000/predict/image" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg"
```

**Example using Python:**
```python
import requests

with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict/image",
        files={"file": f}
    )
print(response.json())
```

---

### Batch Tabular Prediction

#### `POST /predict/batch/tabular`

Make batch predictions on tabular data (optimized for large datasets).

**Request Body:**
```json
{
  "features": [
    [5.1, 3.5, 1.4, 0.2],
    [6.3, 2.9, 5.6, 1.8],
    ...
  ]
}
```

**Response:**
```json
{
  "predictions": [0, 2, 1, ...],
  "num_samples": 100,
  "inference_time_ms": 250.5
}
```

**Note:** Batch prediction returns only class indices (not full prediction objects) for efficiency.

---

## Error Responses

All error responses follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Error Codes

- `400 Bad Request`: Invalid input data
  - Missing required fields
  - Invalid data types
  - Malformed JSON

- `500 Internal Server Error`: Server-side error
  - Model inference failed
  - Unexpected exception

- `503 Service Unavailable`: Service not ready
  - Model not loaded
  - Dependencies unavailable

---

## Rate Limiting

Currently not implemented. Consider adding for production:
- Per-IP rate limits
- API key-based quotas
- Adaptive throttling

---

## Response Times

Expected response times (CPU inference):

| Endpoint | Single Sample | Batch (32) |
|----------|---------------|------------|
| Tabular  | < 50ms        | < 200ms    |
| Image    | < 200ms       | < 2s       |

*Note: GPU inference significantly faster (2-10x)*

---

## Best Practices

### 1. Batch Requests
For multiple predictions, use batch endpoints:
```python
# Good: Single batch request
response = requests.post(url, json={"features": all_samples})

# Bad: Multiple single requests
for sample in all_samples:
    response = requests.post(url, json={"features": [sample]})
```

### 2. Error Handling
Always handle errors gracefully:
```python
try:
    response = requests.post(url, json=data)
    response.raise_for_status()
    result = response.json()
except requests.exceptions.HTTPError as e:
    print(f"HTTP error: {e}")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
```

### 3. Timeouts
Set appropriate timeouts:
```python
response = requests.post(url, json=data, timeout=10)
```

### 4. Input Validation
Validate input before sending:
```python
# Check feature dimensions
assert len(features[0]) == expected_dim

# Check data types
assert all(isinstance(x, (int, float)) for row in features for x in row)
```

---

## SDK Examples

### Python SDK

```python
class ModelClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def predict_tabular(self, features):
        response = requests.post(
            f"{self.base_url}/predict/tabular",
            json={"features": features}
        )
        response.raise_for_status()
        return response.json()
    
    def predict_image(self, image_path):
        with open(image_path, "rb") as f:
            response = requests.post(
                f"{self.base_url}/predict/image",
                files={"file": f}
            )
        response.raise_for_status()
        return response.json()

# Usage
client = ModelClient()
result = client.predict_tabular([[5.1, 3.5, 1.4, 0.2]])
print(result)
```

---

## Testing the API

### Using HTTPie
```bash
# Health check
http GET localhost:8000/health

# Tabular prediction
echo '{"features": [[5.1, 3.5, 1.4, 0.2]]}' | http POST localhost:8000/predict/tabular

# Image prediction
http POST localhost:8000/predict/image file@image.jpg
```

### Using Postman
1. Import the API collection (future: provide Postman collection)
2. Set base URL to `http://localhost:8000`
3. Test endpoints

---

## OpenAPI Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

These provide:
- Interactive endpoint testing
- Request/response schemas
- Example values
- Error codes

---

## Future Enhancements

- [ ] WebSocket support for streaming predictions
- [ ] GraphQL API
- [ ] gRPC support for high-performance
- [ ] API versioning (v1, v2, etc.)
- [ ] Request signing for security
- [ ] Compression support (gzip)
- [ ] CORS configuration
- [ ] OpenAPI 3.1 specification export
