import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent, DragEvent } from 'react'
import './App.css'

type RecipePrediction = {
  title: string
  ingredients: string[]
  instructions: string
  similarity: number
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') ?? ''
const SWAGGER_URL = API_BASE_URL
  ? `${API_BASE_URL}/docs`
  : `${window.location.protocol}//${window.location.hostname}:8000/docs`

const ACCEPTED_FILE_TYPES = ['image/jpeg', 'image/png', 'image/webp']

const quickSteps = [
  'Upload a dish photo from your computer or drag it into the drop area.',
  'The React frontend sends the image to the Python backend with multipart form data.',
  'Get the best recipe match together with confidence, ingredients, and preparation steps.',
]

function App() {
  const inputRef = useRef<HTMLInputElement>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [result, setResult] = useState<RecipePrediction | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dishName, setDishName] = useState('')

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null)
      return undefined
    }

    const nextPreviewUrl = URL.createObjectURL(selectedFile)
    setPreviewUrl(nextPreviewUrl)

    return () => {
      URL.revokeObjectURL(nextPreviewUrl)
    }
  }, [selectedFile])

  const handleFileSelection = (file: File | null) => {
    if (!file) {
      return
    }

    if (!ACCEPTED_FILE_TYPES.includes(file.type)) {
      setError('Unsupported format. Please use JPG, PNG, or WEBP.')
      return
    }

    setSelectedFile(file)
    setResult(null)
    setError(null)
  }

  const onInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    handleFileSelection(event.target.files?.[0] ?? null)
  }

  const onDrop = (event: DragEvent<HTMLButtonElement>) => {
    event.preventDefault()
    setIsDragging(false)
    handleFileSelection(event.dataTransfer.files?.[0] ?? null)
  }

  const analyzeDish = async () => {
    if (!selectedFile) {
      setError('Select an image before starting the analysis.')
      return
    }

    const formData = new FormData()
    formData.append('file', selectedFile)
    if(dishName.trim().length > 0) {
      formData.append('dish_name', dishName.trim())
    }

    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE_URL}/recipes/predict`, {
        method: 'POST',
        body: formData,
      })

      const payload = (await response.json()) as RecipePrediction | { detail?: string }

      if (!response.ok) {
        const detail = 'detail' in payload ? payload.detail : undefined
        throw new Error(detail ?? 'Analysis failed.')
      }

      setResult(payload as RecipePrediction)
    } catch (requestError) {
      const message =
        requestError instanceof Error
          ? requestError.message
          : 'Unexpected error while communicating with the backend.'

      setError(message)
      setResult(null)
    } finally {
      setIsLoading(false)
    }
  }

  const resetExperience = () => {
    setSelectedFile(null)
    setResult(null)
    setError(null)
    setDishName('')  

    if (inputRef.current) {
      inputRef.current.value = ''
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="topbar-brand">
          <span className="topbar-dot"></span>
          <span>What The Food</span>
        </div>
        <a className="topbar-link" href={SWAGGER_URL} target="_blank" rel="noreferrer">
          Open API Swagger
        </a>
      </header>

      <section className="hero-panel">
        <div className="hero-copy">
          <p className="eyebrow">Image to Recipe</p>
          <h1>Discover the recipe behind a dish photo.</h1>
          <p className="hero-text">
            The React frontend sends your image to the Python backend, which compares
            the visual content against the dataset embeddings and returns the most
            likely recipe.
          </p>

          <div className="hero-actions">
            <button type="button" className="primary-button" onClick={() => inputRef.current?.click()}>
              Choose image
            </button>
            <button
              type="button"
              className="secondary-button"
              onClick={analyzeDish}
              disabled={!selectedFile || isLoading}
            >
              {isLoading ? 'Analyzing...' : 'Analyze dish'}
            </button>
          </div>

          <ul className="steps-list">
            {quickSteps.map((step) => (
              <li key={step}>{step}</li>
            ))}
          </ul>
        </div>

        <div className="upload-card">
          <input
            ref={inputRef}
            className="sr-only"
            type="file"
            accept="image/jpeg,image/png,image/webp"
            onChange={onInputChange}
          />

          <button
            type="button"
            className={`dropzone ${isDragging ? 'dragging' : ''}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(event) => {
              event.preventDefault()
              setIsDragging(true)
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={onDrop}
          >
            {previewUrl ? (
              <img className="preview-image" src={previewUrl} alt="Preview of the selected dish" />
            ) : (
              <div className="dropzone-empty">
                <span className="dropzone-icon">Scan</span>
                <strong>Drop your dish photo here</strong>
                <span>or click to browse from your computer</span>
              </div>
            )}
          </button>

          <div className="upload-meta">
            <div>
              <p className="meta-label">Selected file</p>
              <p className="meta-value">{selectedFile ? selectedFile.name : 'No file selected yet'}</p>
            </div>
            <button type="button" className="ghost-button" onClick={resetExperience} disabled={!selectedFile && !result}>
              Reset
            </button>
          </div>

          <div className="dish-name-wrapper">
            <p className="meta-label">If you already know the dish name, type it here:</p>
            <input
              type="text"
              className="dish-name-input"
              value={dishName}
              onChange={(e) => setDishName(e.target.value)}
              disabled={!selectedFile} 
            />
          </div>

          {error ? <p className="status-message error">{error}</p> : null}

          {error ? <p className="status-message error">{error}</p> : null}
          {!error && isLoading ? <p className="status-message">The backend is analyzing your image...</p> : null}
        </div>
      </section>

      <section className="results-grid">
        <article className="result-card spotlight-card">
          <div className="card-header">
            <p className="section-tag">Prediction</p>
            <h2>Model result</h2>
          </div>

          {result ? (
            <div className="prediction-body">
              <div className="score-badge">
                <span>Confidence</span>
                <strong>{Math.round(result.similarity * 100)}%</strong>
              </div>
              <h3>{result.title}</h3>
              <p>
                This match comes from the backend comparing the uploaded image with the
                recipe embedding dataset.
              </p>
            </div>
          ) : (
            <p className="placeholder-copy">
              After the analysis, you will see the recognized dish and its confidence score here.
            </p>
          )}
        </article>

        <article className="result-card">
          <div className="card-header">
            <p className="section-tag">Ingredients</p>
            <h2>Ingredients</h2>
          </div>

          {result ? (
            <ul className="ingredients-list">
              {result.ingredients.map((ingredient) => (
                <li key={ingredient}>{ingredient}</li>
              ))}
            </ul>
          ) : (
            <p className="placeholder-copy">Ingredients will appear here as soon as the backend responds.</p>
          )}
        </article>

        <article className="result-card instructions-card">
          <div className="card-header">
            <p className="section-tag">Instructions</p>
            <h2>Preparation</h2>
          </div>

          {result ? (
            <p className="instructions-text">{result.instructions}</p>
          ) : (
            <p className="placeholder-copy">The full preparation steps will be shown here.</p>
          )}
        </article>
      </section>
    </main>
  )
}

export default App
