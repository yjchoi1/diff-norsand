# Documentation Deployment

This directory contains the Sphinx documentation for the Diff NorSand project.

## GitHub Pages Deployment

The documentation is automatically deployed to GitHub Pages using GitHub Actions. The workflow is triggered on every push to the main/master branch.

### Setup Instructions

1. **Enable GitHub Pages**: 
   - Go to your repository settings on GitHub
   - Navigate to "Pages" in the left sidebar
   - Under "Source", select "GitHub Actions"

2. **Push your changes**: The workflow will automatically run and deploy your documentation

3. **Access your documentation**: After deployment, your docs will be available at:
   `https://[your-username].github.io/[your-repo-name]/`

## Local Development

To build the documentation locally:

```bash
cd docs
pip install -r requirements.txt
make html
```

The built documentation will be in `build/html/`.

## Files

- `source/`: Contains the source files for the documentation
- `build/`: Contains the built HTML files (generated)
- `requirements.txt`: Python dependencies for building the docs
- `Makefile`: Build commands for Unix systems
- `make.bat`: Build commands for Windows 