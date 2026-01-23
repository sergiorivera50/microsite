# Contributing to microsite

## Development Workflow

I use a simple branch-based workflow for development.

### 1. Create a Feature Branch

Always create a new branch for your work:
```bash
# Make sure you're on main and up to date
git checkout main
git pull origin main

# Create and switch to a new branch
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - for new features (e.g., `feature/user-authentication`)
- `fix/` - for bug fixes (e.g., `fix/login-error`)
- `refactor/` - for code refactoring (e.g., `refactor/database-queries`)

### 2. Make Your Changes

```bash
git add .
git commit -m "Added syntax highlighting in code blocks"
```

### 3. Push Your Branch
```bash
git push origin feature/your-feature-name
```

### 4. Create a Pull Request

- Go to the repository on GitHub
- Click "New Pull Request"
- Select your branch
- Add a clear title and description of what changed
- Review changes

### 5. Merge

Once approved:
- Merge the pull request on GitHub
- Delete the feature branch
- Pull the latest main locally:
```bash
git checkout main
git pull origin main
```
