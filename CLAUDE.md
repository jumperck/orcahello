# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.


## Claude Code Workspace Organization

Before proceeding with details about the repo in the next sections, @Claude please read some notes on how to organize your work.

### InferenceSystem

#### Working Directory

Use `InferenceSystem/claude-scratch/` as your working directory for all temporary work, testing, and documentation.

#### Directory Organization

The `claude-scratch/` directory follows this structure:

**Subdirectories**:

- `plans/` - Documentation and planning
  - User-provided task descriptions and requirements (e.g., `<description>.md`)
  - Generated implementation plans (e.g., `task-<number>-<description>.plan.md`)
  - Master plan with tracker of overall tasks and brief per task goals and status

- `results/` - Output documentation
  - Final documentation of completed and in progress tasks
  - Detailed findings (test results, analysis etc.) for completed and in progress tasks
  - Key supporting files (metrics, logs, summaries etc.)
  - Intermediate `CHECKPOINT*.md` files with WIP notes for in progress tasks (that get converted to results files when completed)

- `tmp/` - Temporary files (safe to delete)
  - Scratch space for files that don't need to be preserved
  - Log files from test runs, intermediate outputs etc.
  - Or quick and dirty scripts for investigations
  - Assume it can be cleaned up after task completion without concern

- `test_data/` - Test data
  - Sample WAV files for testing
  - Generated spectrograms (PNG files)
  - Reference audio for validation

**Root Level** (Configuration & Scripts):
- Respective setup instructions to be used for ongoing tasks
- For e.g. `SETUP*.md`, `requirements*.txt`, `Dockerfile` etc.
- Any non-temporary executable scripts and config files for which easy access would help a human end user

#### File Naming Conventions

- **Plans**: `<task-number>-<descriptive_name>.plan.md`
- **Results**: `<component>_results.md` or `<test-name>_summary.md`
- **Scripts**: `<purpose>_<component>.py` (e.g., `test_local_wav.py`)
- **Logs**: `<test-name>_output.log` in `tmp/`

#### Integration with Main Repository

Files in `claude-scratch/` are for temporary planning, documentation and experimentation. When integrating changes:
- Move finalized scripts to appropriate locations in main `InferenceSystem/` structure
- Document permanent changes in main README/docs

#### Example Directory Structure

```
  claude-scratch/
  ├── plans/
  │   ├── master-model_port_hf.plan.md    # Master plan with task tracker
  │   ├── model_port_hf.md                # raw user provided task description (kept as reference)
  │   ├── task-0-*.plan.md                # Task 0 plans
  │   ├── task-2_5-*.plan.md              # Task 2.5 plan
  │   └── ...
  ├── results/
  │   ├── CHECKPOINT_model_v1_port.md     # Current checkpoint
  │   ├── task4_full_file_inference.md    # Task 4 results
  │   └── ...
  ├── SETUP_INFERENCE_VENV.md             # Environment setup docs
  └── SETUP_MODEL_V1_VENV.md
```


## Project Overview

OrcaHello is a real-time AI-assisted killer whale notification system maintained by Orcasound. The system uses deep learning to detect orca calls in live hydrophone audio streams from Puget Sound, with the goal of helping recover the endangered Southern Resident Killer Whale (SRKW) population.

## System Architecture

The system processes audio data through these stages:

1. **Live streaming**: Audio streams from AWS S3 buckets (Orcasound's hydrophone network)
2. **Azure-based analysis**: Ingestion of 10-second segments, inference on 2-second samples, concatenation into 60-second WAV files and spectrogram generation
3. **Moderation**: Expert review of model detections via the moderator portal
4. **Notification**: Email alerts to subscribers when confirmed SRKW calls are detected

### Component Architecture

- **InferenceSystem**: Streams audio from S3, performs model inference, uploads detections to Azure
  - Entry point: `InferenceSystem/src/LiveInferenceOrchestrator.py`
  - Deployed via Azure Kubernetes Service (AKS)
  - Each hydrophone location runs in its own namespace

- **ModelTraining**: Data preparation and model training using FastAI
  - Transfer learning with ResNet18 on spectrogram images
  - Current production model: `11-15-20.FastAI.R1-12`
  - Training notebooks use Jupyter

- **ModeratorFrontEnd**: Web portal for expert review of detections. Two generations coexist:
  - **Gen 1 — `AIForOrcas`** (.NET 6, current production): `ModeratorFrontEnd/AIForOrcas/`
    - `AIForOrcas.Client.Web` (Blazor) → Azure App Service `AIForOrcas` → `aifororcas.azurewebsites.net`
    - `AIForOrcas.Server` (Web API) → Azure App Service `AIForOrcasDetections` → `aifororcasdetections.azurewebsites.net`
    - Reads/writes Cosmos DB container `metadata` (partition key `/source_guid`)
  - **Gen 2 — `OrcaHello`** (.NET 8, future replacement): `ModeratorFrontEnd/OrcaHello/`
    - `OrcaHello.Web.UI` (Blazor) → Azure App Service `AIForOrcas2`
    - `OrcaHello.Web.Api` (Web API) → Azure App Service `AIForOrcasDetections2`
    - Reads/writes Cosmos DB container `orca_sounds` (partition key `/state`, migrated schema)
    - Includes `OrcaHello.Console.DataMigration` tool for migrating data from `metadata` → `orca_sounds`

- **NotificationSystem**: Azure Functions (`orcanotification`) for email notifications
  - Triggers on CosmosDB change feed from `metadata` container (uses `leases` container for tracking)
  - Moderator/subscriber email lists stored in Azure Table Storage (`orcanotificationstorage` → Table `EmailList`, partitioned by `"Moderator"` / `"Subscriber"`)
  - Uses AWS Simple Email Service for sending


### Data Flow

Detection metadata flows through: InferenceSystem → CosmosDB → NotificationSystem → Moderators → ModeratorFrontEnd → Validated detections → Subscriber notifications

Audio/spectrogram files are stored in Azure Blob Storage.


## Development Commands

### InferenceSystem (Python)

```bash
cd InferenceSystem

# Create virtual environment (Python 3.11 recommended)
python -m venv inference-venv
source inference-venv/bin/activate  # Mac/Linux
# .\inference-venv\Scripts\activate.bat  # Windows

# Run inference locally (test mode)
python src/LiveInferenceOrchestrator.py --config ./config/Test/FastAI_LiveHLS_OrcasoundLab.yml

# Run with limited iterations for testing
python src/LiveInferenceOrchestrator.py --config ./config/Test/FastAI_LiveHLS_OrcasoundLab.yml --max_iterations 2
```

For more details on setup, see `InferenceSystem/claude-scratch/SETUP.md`

**Dependencies:**
- Python 3.11
- FFmpeg (install via `apt-get install ffmpeg` on Linux or `choco install ffmpeg` on Windows)

**Required environment variables** (not needed when `upload_to_azure: False` in test mode):
- `AZURE_STORAGE_CONNECTION_STRING`
- `AZURE_COSMOSDB_PRIMARY_KEY`
- `INFERENCESYSTEM_APPINSIGHTS_CONNECTION_STRING`

### ModelTraining (Python + Jupyter)

```bash
cd ModelTraining

# Create conda environment
conda create -n <env-name> python=3.8
conda activate <env-name>

# Install dependencies (using uv for better version resolution)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install -r requirements.txt

# Start Jupyter and open notebooks:
# - 1_DataCreation.ipynb: Prepare training data
# - 2_FastAI_StarterScript.ipynb: Train model
# - 3_InferenceTesting.ipynb: Test inference
```

### NotificationSystem (.NET 8)

```bash
cd NotificationSystem/NotificationSystem

# Build from command line
dotnet build NotificationSystem.csproj

# Run unit tests
cd ../NotificationSystem.Tests.Unit
dotnet test

# Run integration tests
cd ../NotificationSystem.Tests.Integration
dotnet test

# Run locally (requires local.settings.json with Azure connection strings)
cd ../NotificationSystem
func start
```

### ModeratorFrontEnd — Gen 1: AIForOrcas (.NET 6)

```bash
cd ModeratorFrontEnd/AIForOrcas/AIForOrcas.Server
dotnet restore && dotnet build -c Release
dotnet test
```

### ModeratorFrontEnd — Gen 2: OrcaHello (.NET 8)

```bash
cd ModeratorFrontEnd/OrcaHello/OrcaHello.Web.Api
dotnet restore && dotnet build -c Release
dotnet test

cd ../OrcaHello.Web.UI
dotnet restore && dotnet build -c Release
dotnet test
```



## Azure Resources

All resources are in the Azure subscription under `LiveSRKWNotificationSystem` resource group:

- **AKS**: `inference-system-AKS` (InferenceSystem, one namespace per hydrophone)
- **Container Registry**: `orcaconservancycr.azurecr.io`
- **App Services (Gen 1)**: `AIForOrcas` (Blazor frontend), `AIForOrcasDetections` (Web API)
- **App Services (Gen 2)**: `AIForOrcas2` (OrcaHello frontend), `AIForOrcasDetections2` (OrcaHello API)
- **Azure Functions**: `orcanotification` (NotificationSystem)
- **Monitoring**: `InferenceSystemInsights` (App Insights)

### Storage Accounts

- **`livemlaudiospecstorage`**: Audio WAV files and spectrogram images from InferenceSystem
- **`orcanotificationstorage`**: Notification-related storage
  - Table `EmailList`: Moderator and subscriber emails (partition keys `"Moderator"` / `"Subscriber"`)
  - Queue `srkwfound`: Confirmed detections queued for subscriber notification
  - Blob `images/`: Map images used in email templates

### Cosmos DB (`aifororcasmetadatastore`)

Single account, single database `predictions`, with these containers:

| Container | Used by | Partition key | Purpose |
|---|---|---|---|
| `metadata` | Gen 1 (`AIForOrcas`), InferenceSystem, NotificationSystem | `/source_guid` | Active production detection metadata |
| `orca_sounds` | Gen 2 (`OrcaHello`) | `/state` | Migrated detection data with updated schema |
| `leases` | NotificationSystem | — | Change feed tracking (prefixes: `moderator`, `orcasite`, `subscriber`) |
| `metadatabystate` | Unused | — | Likely an intermediate migration artifact |

**Schema differences** (`metadata` → `orca_sounds`): `reviewed` + `SRKWFound` consolidated into `state` (`Unreviewed`/`Positive`/`Negative`/`Unknown`); `tags` changed from semicolon-delimited string to `List<string>`; added top-level `locationName`; dropped `source_guid`.

## CI/CD and Deployment

### Workflow Summary

| Workflow file | Trigger | Auto-deploys? | Azure target |
|---|---|---|---|
| `InferenceSystem.yaml` | PR + push to `main` | No (CI only) | — |
| `InferenceSystem-deploy.yaml` | Git tag `InferenceSystem.v*` or manual | To ACR only | `orcaconservancycr.azurecr.io` (AKS deploy is manual `kubectl apply`) |
| `AIForOrcas.Client.Web.yaml` | Push to `main` (path-filtered) | **Yes** | App Service `AIForOrcas` → `aifororcas.azurewebsites.net` |
| `AIForOrcas.Server.yaml` | Push to `main` (path-filtered) | **Yes** | App Service `AIForOrcasDetections` → `aifororcasdetections.azurewebsites.net` |
| `OrcaHello.Web.UI.yaml` | Push to `main` (path-filtered) | **Yes** | App Service `AIForOrcas2` |
| `OrcaHello.Web.Api.yaml` | Push to `main` (path-filtered) | **Yes** | App Service `AIForOrcasDetections2` |
| `NotificationSystem.yaml` | Push to `main` (path-filtered) | **Yes** | Azure Functions `orcanotification` |
| `InferenceSystem-deploy-configmaps.yaml` | Manual | No | AKS configmaps |
| `validate-yaml.yml` | PR/push | No (lint only) | — |
| `scorecard.yml` | Scheduled | No (security audit) | — |

All auto-deploy workflows use `dorny/paths-filter` to only build/deploy when relevant source files change.

### InferenceSystem

#### CI (InferenceSystem.yaml)

Runs on PRs and pushes to main. Tests on Windows + Ubuntu with Python 3.11. Downloads production model `11-15-20.FastAI.R1-12`, tests all hydrophone locations (positive, negative, and error cases), and builds/tests the Docker container.

#### Deployment (InferenceSystem-deploy.yaml)

Triggered by git tags matching `InferenceSystem.v[0-9]+.[0-9]+.[0-9]+` or manual dispatch. Builds Docker image → pushes to ACR. AKS deployment is a separate manual step:

```bash
git tag InferenceSystem.v1.0.0
git push origin InferenceSystem.v1.0.0

# After image is in ACR, manually deploy to AKS:
az aks get-credentials -g LiveSRKWNotificationSystem -n inference-system-AKS
kubectl apply -f deploy/bush-point.yaml
```

Docker image versioning: `orcaconservancycr.azurecr.io/live-inference-system:<MM-DD-YYYY>.FastAI.R1-12.v<major>`

### ModeratorFrontEnd

Four workflows cover the two generations. All auto-deploy on push to `main` when relevant files change:

- **`AIForOrcas.Client.Web.yaml`** — .NET 6, deploys Blazor app → App Service `AIForOrcas`
- **`AIForOrcas.Server.yaml`** — .NET 6, deploys Web API → App Service `AIForOrcasDetections`
- **`OrcaHello.Web.UI.yaml`** — .NET 8, deploys Blazor app → App Service `AIForOrcas2`
- **`OrcaHello.Web.Api.yaml`** — .NET 8, deploys Web API → App Service `AIForOrcasDetections2`

### NotificationSystem (NotificationSystem.yaml)

.NET 8, tests on Ubuntu (unit) + Windows (unit + integration). Auto-deploys on push to `main` → Azure Functions `orcanotification`.


## Key Technical Details

- **Model**: FastAI ResNet18 trained on spectrograms, performs binary classification (whale call / not whale call) on 2-second audio segments
- **Audio Processing**: 10-second HLS segments from S3 → concatenated into 60-second WAV files → resampled to target rate → converted to spectrograms
- **Detection Logic**: Overlapping 2-second segments classified; adjacent positive detections are concatenated
- **Hydrophone Locations**: Each hydrophone has its own config file and can be deployed independently
- **Authentication**: ModeratorFrontEnd uses Azure Active Directory for policy-based authentication

## Important File Patterns

- `InferenceSystem/config/*.yml`: Hydrophone configuration files
  - `InferenceSystem/config/Test/FastAI_LiveHLS_OrcasoundLab.yml`: Live HLS test configuration
  - `InferenceSystem/config/Test/Positive/*.yml`: Test configs that should detect whale calls
  - `InferenceSystem/config/Test/Negative/*.yml`: Test configs that should not detect whale calls
  - `InferenceSystem/config/Test/Fail/*.yml`: Test configs for error handling (NoAudio, IncompleteMinute)
- `InferenceSystem/deploy/*.yaml`: Kubernetes deployment manifests
- `InferenceSystem/patch_fastai_audio.sh/bat`: Patches fastai_audio for Python 3.11+ compatibility
- `ModelTraining/*.ipynb`: Jupyter notebooks for model development
- `.github/workflows/InferenceSystem.yaml`: InferenceSystem CI (tests only)
- `.github/workflows/InferenceSystem-deploy.yaml`: InferenceSystem deployment to ACR (tag-triggered)
- `.github/workflows/AIForOrcas.Server.yaml`: Gen 1 Web API CI/CD → `AIForOrcasDetections`
- `.github/workflows/AIForOrcas.Client.Web.yaml`: Gen 1 Blazor app CI/CD → `AIForOrcas`
- `.github/workflows/OrcaHello.Web.Api.yaml`: Gen 2 Web API CI/CD → `AIForOrcasDetections2`
- `.github/workflows/OrcaHello.Web.UI.yaml`: Gen 2 Blazor app CI/CD → `AIForOrcas2`
- `.github/workflows/NotificationSystem.yaml`: NotificationSystem CI/CD → Azure Functions `orcanotification`


## Working with Hydrophones

To add a new hydrophone location:

1. Create config file in `InferenceSystem/config/`
2. Update `InferenceSystem/src/LiveInferenceOrchestrator.py` and `src/globals.py` with new location variables
3. Create deployment YAML in `InferenceSystem/deploy/`
4. Create namespace and secret in AKS:
   ```bash
   kubectl create namespace <location-name>
   kubectl create secret generic inference-system -n <location-name> \
       --from-literal=AZURE_COSMOSDB_PRIMARY_KEY='<key>' \
       --from-literal=AZURE_STORAGE_CONNECTION_STRING='<string>' \
       --from-literal=INFERENCESYSTEM_APPINSIGHTS_CONNECTION_STRING='<string>'
   ```
5. Deploy: `kubectl apply -f deploy/<location-name>.yaml`


## Data Sources

- Training data: Hosted on [Orcasound Pod.Cast archive](https://github.com/orcasound/orcadata/wiki/Pod.Cast-data-archive)
- Live audio: Orcasound S3 buckets (e.g., `s3://audio-orcasound-net/rpi_orcasound_lab`)
- Test sets: OrcasoundLab07052019_Test, OrcasoundLab09272017_Test
