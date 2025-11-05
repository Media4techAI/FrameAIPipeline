# Diagrammi PlantUML - Frame Pipeline

Questa cartella contiene i diagrammi PlantUML che documentano l'architettura del Frame Pipeline System.

## Struttura Diagrammi

### Architettura Sistema
- **`high_level_architetture.puml`** - Architettura generale del sistema con tutti i layer
- **`component_diagram.puml`** - Diagramma dei componenti software (C4 model)

### Flusso Dati e Processo
- **`dataflow_architetture.puml`** - Flusso completo dei dati nel sistema
- **`dataflow.puml`** - Pipeline core flow step-by-step
- **`data_transformation.puml`** - Trasformazioni dei frammenti di immagine
- **`sequence_diagram.puml`** - Sequenza temporale di esecuzione

### Architettura Filtri
- **`chain_architetture.puml`** - Architettura della catena di filtri
- **`new_filters.puml`** - Estensibilità e nuovi filtri

## Rendering dei Diagrammi

I diagrammi possono essere renderizzati usando:

### VS Code (Raccomandato)
1. Installa l'estensione "PlantUML"
2. Apri qualsiasi file `.puml`
3. Premi `Alt+D` per preview

### Online
- [PlantUML Online Server](http://www.plantuml.com/plantuml/uml/)
- [PlantText](https://www.planttext.com/)

### Command Line
```bash
# Installa PlantUML
sudo apt install plantuml  # Ubuntu/Debian
brew install plantuml      # macOS

# Genera immagini PNG
plantuml docs/diagrams/*.puml -o ../img/

# Genera immagini SVG
plantuml docs/diagrams/*.puml -tsvg -o ../img/
```

## Immagini Renderizzate

Le immagini renderizzate si trovano in [`../img/`](../img/) e sono referenziate nella documentazione:

- `high_level_architetture.png`
- `component_diagram.png`
- `dataflow_architetture.png`
- `dataflow.png`
- `data_transformation.png`
- `sequence_diagram.png`
- `chain_architetture.png`
- `new_filters.png`

## Aggiornamenti

Quando si modifica un diagramma PlantUML:

1. Aggiorna il file `.puml`
2. Rigenera l'immagine corrispondente
3. Verifica che la documentazione referenzi correttamente il diagramma

## Convenzioni

- **Nomi file**: Usa underscore `_` invece di trattini
- **Colori**: Mantieni coerenza nei colori tra diagrammi simili
- **Note**: Aggiungi note esplicative dove necessario
- **Titoli**: Ogni diagramma deve avere un titolo chiaro

## Dipendenze

Per diagrammi C4:
```plantuml
!include <C4/C4_Component>
```

Per stili personalizzati:
```plantuml
skinparam packageStyle rectangle
skinparam componentStyle rectangle
```