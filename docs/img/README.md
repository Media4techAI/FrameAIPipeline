# Immagini Diagrammi - Frame Pipeline

Questa cartella contiene le immagini renderizzate dei diagrammi PlantUML che documentano l'architettura del Frame Pipeline System.

## Diagrammi Disponibili

### Architettura Sistema
- **`high_level_architetture.png`** - Architettura generale del sistema
  - Sorgente: [`../diagrams/high_level_architetture.puml`](../diagrams/high_level_architetture.puml)
  - Mostra i layer: Input, Processing, Storage, External Components

- **`component_diagram.png`** - Diagramma dei componenti software
  - Sorgente: [`../diagrams/component_diagram.puml`](../diagrams/component_diagram.puml)
  - Architettura C4: componenti, relazioni, dipendenze

### Flusso Dati
- **`dataflow_architetture.png`** - Flusso completo dei dati
  - Sorgente: [`../diagrams/dataflow_architetture.puml`](../diagrams/dataflow_architetture.puml)
  - Mostra attori, processi, data store

- **`dataflow.png`** - Pipeline core flow
  - Sorgente: [`../diagrams/dataflow.puml`](../diagrams/dataflow.puml)
  - Flusso step-by-step con input/output dettagliati

- **`data_transformation.png`** - Trasformazioni frammenti
  - Sorgente: [`../diagrams/data_transformation.puml`](../diagrams/data_transformation.puml)
  - Trasformazione dati da input a output finale

### Processo Esecuzione
- **`sequence_diagram.png`** - Sequenza temporale esecuzione
  - Sorgente: [`../diagrams/sequence_diagram.puml`](../diagrams/sequence_diagram.puml)
  - Interazione temporale tra componenti

### Architettura Filtri
- **`chain_architetture.png`** - Catena di filtri
  - Sorgente: [`../diagrams/chain_architetture.puml`](../diagrams/chain_architetture.puml)
  - Interfacce e implementazioni filtri

- **`new_filters.png`** - Estensibilità sistema
  - Sorgente: [`../diagrams/new_filters.puml`](../diagrams/new_filters.puml)
  - Come aggiungere nuovi filtri

### Altri Diagrammi
- **`boostdag.png`** - Diagramma specifico boost filter

## Formato e Qualità

- **Formato**: PNG con trasparenza
- **Risoluzione**: Alta qualità per documentazione
- **Dimensioni**: Ottimizzate per visualizzazione web/PDF

## Aggiornamento Immagini

Per rigenerare le immagini dai sorgenti PlantUML:

```bash
# Dalla root del progetto
cd docs/diagrams

# Genera tutte le immagini
plantuml *.puml -o ../img/

# Genera immagine specifica
plantuml high_level_architetture.puml -o ../img/
```

## Utilizzo nella Documentazione

Le immagini sono referenziate nella documentazione Markdown:

```markdown
![High Level Architecture](img/high_level_architetture.png)
```

E collegate ai sorgenti PlantUML:

```markdown
> **Diagrammi**: 
> - PlantUML: [high_level_architetture.puml](diagrams/high_level_architetture.puml)
> - Immagine: ![High Level Architecture](img/high_level_architetture.png)
```

## Manutenzione

- Mantieni sincronizzazione tra file `.puml` e immagini `.png`
- Aggiorna immagini quando modifichi i diagrammi sorgente
- Verifica che i link nella documentazione funzionino correttamente