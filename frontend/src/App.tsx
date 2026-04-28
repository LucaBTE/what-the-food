import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent, DragEvent } from 'react'
import './App.css'
import logo from '../assets/wtflogo.png'

type RecipePrediction = {
  title: string
  ingredients: string[]
  instructions: string
  similarity: number
  reference_image_name: string | null
}

type IngredientAlert = 'gluten' | 'lactose' | 'both' | 'none'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') ?? ''
const ACCEPTED_FILE_TYPES = ['image/jpeg', 'image/png', 'image/webp']
const GLUTEN_KEYWORDS = [
  'flour',
  'farina',
  'wheat',
  'grano',
  'semolina',
  'semola',
  'pasta',
  'spaghetti',
  'macaroni',
  'noodle',
  'noodles',
  'bread',
  'pane',
  'breadcrumb',
  'breadcrumbs',
  'pangrattato',
  'cracker',
  'biscuit',
  'cookie',
  'cake',
  'pizza',
  'focaccia',
  'barley',
  'orzo',
  'rye',
  'segale',
  'couscous',
  'bulgur',
  'seitan',
  'beer',
  'birra',
]
const LACTOSE_KEYWORDS = [
  'milk',
  'latte',
  'butter',
  'burro',
  'cream',
  'panna',
  'cheese',
  'formaggio',
  'mozzarella',
  'parmesan',
  'parmigiano',
  'pecorino',
  'ricotta',
  'gorgonzola',
  'mascarpone',
  'yogurt',
  'yoghurt',
  'gelato',
  'ice cream',
  'whey',
  'casein',
  'lactose',
]

const normalizeIngredient = (ingredient: string) => ingredient.toLowerCase()

const hasKeyword = (ingredient: string, keywords: string[]) =>
  keywords.some((keyword) => ingredient.includes(keyword))

const getIngredientAlert = (ingredient: string): IngredientAlert => {
  const normalizedIngredient = normalizeIngredient(ingredient)
  const hasGluten = hasKeyword(normalizedIngredient, GLUTEN_KEYWORDS)
  const hasLactose = hasKeyword(normalizedIngredient, LACTOSE_KEYWORDS)

  if (hasGluten && hasLactose) {
    return 'both'
  }

  if (hasGluten) {
    return 'gluten'
  }

  if (hasLactose) {
    return 'lactose'
  }

  return 'none'
}

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
      setError('Formato non supportato. Usa JPG, PNG o WEBP.')
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

  const getReferenceImageUrl = (imageName: string | null) => {
    if (!imageName) {
      return null
    }

    const encodedName = encodeURIComponent(imageName)
    const baseUrl = API_BASE_URL || ''
    return `${baseUrl}/recipes/reference-image/${encodedName}`
  }

  const analyzeDish = async () => {
    if (!selectedFile) {
      setError("Seleziona un'immagine prima di iniziare.")
      return
    }

    const formData = new FormData()
    formData.append('file', selectedFile)

    if (dishName.trim().length > 0) {
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
        throw new Error(detail ?? 'Analisi non riuscita.')
      }

      setResult(payload as RecipePrediction)
    } catch (requestError) {
      const message =
        requestError instanceof Error
          ? requestError.message
          : 'Errore imprevisto durante la comunicazione con il backend.'

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

  const referenceImageUrl = getReferenceImageUrl(result?.reference_image_name ?? null)
  const ingredientAlerts = result?.ingredients.map((ingredient) => ({
    name: ingredient,
    alert: getIngredientAlert(ingredient),
  }))
  const hasGluten = ingredientAlerts?.some((item) => item.alert === 'gluten' || item.alert === 'both') ?? false
  const hasLactose = ingredientAlerts?.some((item) => item.alert === 'lactose' || item.alert === 'both') ?? false

  return (
    <main className="page-shell">
      <section className="brand-stage" aria-label="What The Food">
        <img className="brand-stage-logo" src={logo} alt="Logo What The Food" />
      </section>

      <section className="app-frame">
        <div className="workspace-grid">
          <section className="composer-column">
            <div className="section-heading">
              <div>
                <p className="section-kicker">Input</p>
                <h2>Carica il tuo piatto</h2>
              </div>
              <button
                type="button"
                className="ghost-button"
                onClick={resetExperience}
                disabled={!selectedFile && !result}
              >
                Reset
              </button>
            </div>

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
                <img className="preview-image" src={previewUrl} alt="Anteprima del piatto selezionato" />
              ) : (
                <div className="dropzone-empty">
                  <span className="dropzone-badge">Drop image</span>
                  <strong>Trascina qui la foto oppure tocca per sceglierla.</strong>
                  <p>Formati supportati: JPG, PNG, WEBP.</p>
                </div>
              )}
            </button>

            <div className="control-strip">
              <label className="dish-input-group">
                <span>Nome del piatto opzionale</span>
                <input
                  type="text"
                  className="dish-name-input"
                  value={dishName}
                  onChange={(event) => setDishName(event.target.value)}
                  disabled={!selectedFile}
                  placeholder="Es. lasagna"
                />
              </label>

              <button
                type="button"
                className="primary-button"
                onClick={analyzeDish}
                disabled={!selectedFile || isLoading}
              >
                {isLoading ? 'Analisi in corso...' : 'Analizza immagine'}
              </button>
            </div>

            <div className="inline-meta">
              <span className="meta-label">File</span>
              <strong>{selectedFile ? selectedFile.name : 'Nessun file selezionato'}</strong>
            </div>

            {error ? <p className="status-message error">{error}</p> : null}
            {!error && isLoading ? (
              <p className="status-message">Sto confrontando immagine e testo con il dataset...</p>
            ) : null}
          </section>

          <section className="result-column">
            <div className="result-topline">
              <div>
                <p className="section-kicker">Best match</p>
                <h2>{result ? result.title : 'In attesa del match migliore'}</h2>
              </div>
              <div className="confidence-pill">
                <span>Affinità</span>
                <strong>{result ? `${Math.round(result.similarity * 100)}%` : '--'}</strong>
              </div>
            </div>

            <div className="match-stage">
              <article className="image-slot">
                <div className="image-slot-label">La tua immagine</div>
                {previewUrl ? (
                  <img className="stage-image" src={previewUrl} alt="Foto caricata" />
                ) : (
                  <div className="image-placeholder">La preview della tua foto apparira qui.</div>
                )}
              </article>

              <article className="image-slot reference">
                <div className="image-slot-label">Immagine più simile</div>
                {referenceImageUrl ? (
                  <img
                    className="stage-image"
                    src={referenceImageUrl}
                    alt={result ? `Immagine di riferimento per ${result.title}` : 'Immagine di riferimento'}
                  />
                ) : (
                  <div className="image-placeholder">
                    Dopo l’analisi vedrai qui l’immagine esatta del dataset usata come match.
                  </div>
                )}
              </article>
            </div>

            <div className="result-copy">

              <div className="content-flow">
                <section className="flow-block">
                  <p className="section-kicker">Ingredienti</p>
                  {result ? (
                    <>
                      {hasGluten || hasLactose ? (
                        <div className="dietary-alerts" role="status" aria-live="polite">
                          {hasGluten ? (
                            <div className="dietary-alert dietary-alert-gluten">
                              <strong>Attenzione glutine</strong>
                              <span>La ricetta contiene ingredienti che possono contenere glutine.</span>
                            </div>
                          ) : null}
                          {hasLactose ? (
                            <div className="dietary-alert dietary-alert-lactose">
                              <strong>Attenzione lattosio</strong>
                              <span>La ricetta contiene ingredienti che possono contenere lattosio.</span>
                            </div>
                          ) : null}
                        </div>
                      ) : (
                        <div className="dietary-alerts" role="status" aria-live="polite">
                          <div className="dietary-alert dietary-alert-safe">
                            <strong>Nessun segnale evidente</strong>
                            <span>
                              Non ho rilevato ingredienti chiaramente associati a glutine o lattosio.
                            </span>
                          </div>
                        </div>
                      )}

                      <div className="ingredient-cloud">
                        {ingredientAlerts?.map(({ name, alert }) => (
                          <span
                            key={name}
                            className={`ingredient-pill ingredient-pill-${alert}`}
                            title={
                              alert === 'gluten'
                                ? 'Possibile presenza di glutine'
                                : alert === 'lactose'
                                  ? 'Possibile presenza di lattosio'
                                  : alert === 'both'
                                    ? 'Possibile presenza di glutine e lattosio'
                                    : 'Nessun allergene rilevato'
                            }
                          >
                            <span>{name}</span>
                            {alert === 'gluten' ? <strong>Glutine</strong> : null}
                            {alert === 'lactose' ? <strong>Lattosio</strong> : null}
                            {alert === 'both' ? <strong>Glutine + Lattosio</strong> : null}
                          </span>
                        ))}
                      </div>
                    </>
                  ) : (
                    <p className="placeholder-copy">Gli ingredienti appariranno qui dopo il match.</p>
                  )}
                </section>

                <section className="flow-block">
                  <p className="section-kicker">Procedimento</p>
                  {result ? (
                    <p className="instructions-text">{result.instructions}</p>
                  ) : (
                    <p className="placeholder-copy">Il procedimento completo comparira qui.</p>
                  )}
                </section>
              </div>
            </div>
          </section>
        </div>
      </section>
    </main>
  )
}

export default App
