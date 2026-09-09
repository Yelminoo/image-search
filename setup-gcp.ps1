<#
.SYNOPSIS
  One-time GCP provisioning for the OCR + Vector Search stack.

.DESCRIPTION
  Runs steps 1-9 from the setup walkthrough: project selection, billing check,
  API enablement, GCS bucket, Firestore database, service account + IAM
  bindings, key creation, and a final sanity check.

  Does NOT run setup_vector_index.py (that's a separate 20-60 min step, run
  after this script finishes and .env is filled in).

.USAGE
  1. Edit the "EDIT ME" block below.
  2. Open PowerShell in this folder.
  3. .\setup-gcp.ps1
#>

$ErrorActionPreference = "Stop"

# ============ EDIT ME ============
$ProjectId          = "your-gcp-project-id"       # must be globally unique if creating new
$CreateNewProject   = $false                       # $true to create $ProjectId, $false to use an existing one
$BillingAccountId   = ""                           # only needed if CreateNewProject or billing isn't linked yet; find with: gcloud billing accounts list
$Region             = "us-central1"                # used for GCS bucket + Firestore + Vertex AI location
$BucketName         = "your-bucket-name"           # must be globally unique
$ServiceAccountName = "ocr-search-sa"
$KeyOutputPath      = ".\service-account.json"
# ==================================

function Step($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

function TryRun($scriptBlock, $ignoreIfExists = $true) {
    try {
        & $scriptBlock
    } catch {
        if ($ignoreIfExists -and $_.Exception.Message -match "already exists|ALREADY_EXISTS") {
            Write-Host "    (already exists, skipping)" -ForegroundColor DarkYellow
        } else {
            throw
        }
    }
}

# --- 1. Auth check ---
Step "Checking authenticated account"
$account = gcloud config get-value account 2>$null
if (-not $account -or $account -eq "(unset)") {
    Write-Host "No active gcloud auth. Running 'gcloud auth login'..."
    gcloud auth login
} else {
    Write-Host "Authenticated as: $account"
}

# --- 2. Project ---
Step "Setting up project: $ProjectId"
if ($CreateNewProject) {
    TryRun { gcloud projects create $ProjectId --name=$ProjectId }
}
gcloud config set project $ProjectId

# --- 3. Billing ---
Step "Checking billing status"
$billingInfo = gcloud billing projects describe $ProjectId --format="value(billingEnabled)" 2>$null
if ($billingInfo -ne "True") {
    if (-not $BillingAccountId) {
        Write-Host "Billing is NOT enabled and no BillingAccountId was set in this script." -ForegroundColor Red
        Write-Host "Run 'gcloud billing accounts list', put the ID in `$BillingAccountId` above, and re-run." -ForegroundColor Red
        exit 1
    }
    Write-Host "Linking billing account $BillingAccountId..."
    gcloud billing projects link $ProjectId --billing-account=$BillingAccountId
} else {
    Write-Host "Billing already enabled."
}

# --- 4. Enable APIs ---
Step "Enabling required APIs (Vision, Vertex AI, Storage, Firestore)"
gcloud services enable `
    vision.googleapis.com `
    aiplatform.googleapis.com `
    storage.googleapis.com `
    firestore.googleapis.com `
    --project=$ProjectId

# --- 5. GCS bucket ---
Step "Creating GCS bucket: gs://$BucketName"
TryRun { gcloud storage buckets create "gs://$BucketName" --location=$Region --uniform-bucket-level-access --project=$ProjectId }

# --- 6. Firestore database ---
Step "Creating Firestore database (Native mode) in $Region"
Write-Host "NOTE: this can only be created ONCE per project and cannot be changed later." -ForegroundColor Yellow
TryRun { gcloud firestore databases create --location=$Region --type=firestore-native --project=$ProjectId }

# --- 7. Service account ---
$saEmail = "$ServiceAccountName@$ProjectId.iam.gserviceaccount.com"
Step "Creating service account: $saEmail"
TryRun { gcloud iam service-accounts create $ServiceAccountName --display-name="OCR + Vector Search app" --project=$ProjectId }

# --- 8. IAM bindings ---
Step "Granting IAM roles to $saEmail"
$roles = @("roles/aiplatform.user", "roles/storage.objectAdmin", "roles/datastore.user")
foreach ($role in $roles) {
    Write-Host "  binding $role"
    gcloud projects add-iam-policy-binding $ProjectId `
        --member="serviceAccount:$saEmail" `
        --role=$role `
        --condition=None `
        --quiet | Out-Null
}

# --- 9. Verify bindings ---
Step "Verifying IAM bindings"
gcloud projects get-iam-policy $ProjectId `
    --flatten="bindings[].members" `
    --filter="bindings.members:$saEmail" `
    --format="table(bindings.role)"

# --- 10. Create key ---
Step "Creating service account key -> $KeyOutputPath"
if (Test-Path $KeyOutputPath) {
    Write-Host "  $KeyOutputPath already exists, not overwriting. Delete it first if you want a fresh key." -ForegroundColor DarkYellow
} else {
    gcloud iam service-accounts keys create $KeyOutputPath --iam-account=$saEmail
}

# --- 11. Sanity check ---
Step "Sanity-checking the service account can get a token"
gcloud auth application-default print-access-token --impersonate-service-account=$saEmail | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "Token acquired OK." -ForegroundColor Green
} else {
    Write-Host "Token check failed - see error above." -ForegroundColor Red
}

Step "Done. Next steps:"
Write-Host @"
1. cp .env.example .env   (or copy it manually on Windows)
2. Fill in .env with:
     GCP_PROJECT_ID=$ProjectId
     GCP_LOCATION=$Region
     GCS_BUCKET=$BucketName
     GOOGLE_APPLICATION_CREDENTIALS=$KeyOutputPath
3. pip install -r requirements.txt
4. python setup_vector_index.py   (takes 20-60 min, prints VERTEX_INDEX_ID / VERTEX_INDEX_ENDPOINT_ID)
5. Paste those two IDs into .env
6. uvicorn app.main:app --reload --port 8000
"@
