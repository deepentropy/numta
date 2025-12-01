# Release Process

This project uses **setuptools-scm** for automatic version management from git tags. You don't need to manually update version numbers in the code.

## How It Works

The version is automatically derived from git tags:
- **Tagged release**: `v0.1.1` → version `0.1.1`
- **Development**: `v0.1.1` + 5 commits → version `0.1.1.dev5+g<hash>`
- **No tags**: version `0.0.0+unknown`

## Creating a New Release

### 1. Ensure all changes are committed and pushed

```bash
git add .
git commit -m "Your changes"
git push origin main
```

### 2. Create and push a git tag

```bash
# For version 0.1.2
git tag v0.1.2
git push origin v0.1.2
```

Or use an annotated tag (recommended):

```bash
git tag -a v0.1.2 -m "Release version 0.1.2"
git push origin v0.1.2
```

### 3. Create a GitHub Release

1. Go to https://github.com/deepentropy/numta/releases
2. Click "Draft a new release"
3. Select the tag you just created (e.g., `v0.1.2`)
4. Fill in the release notes
5. Click "Publish release"

### 4. Automatic Publishing

The GitHub Actions workflow will automatically:
1. Checkout the code at the tagged commit
2. Detect the version from the git tag using setuptools-scm
3. Build the package with the correct version
4. Publish to PyPI

**No manual version updates needed!** ✨

## Version Format

We follow [Semantic Versioning](https://semver.org/):
- **v0.1.0** → Initial release
- **v0.1.1** → Bug fix
- **v0.2.0** → New features (backward compatible)
- **v1.0.0** → Major release (breaking changes)

## Checking the Current Version

### From git (without building):
```bash
python -m setuptools_scm
```

### After installation:
```python
import numta
print(numta.__version__)
```

## Development Versions

Between releases, the version will automatically include:
- Number of commits since last tag
- Git commit hash
- Example: `0.1.2.dev5+g1a2b3c4`

This ensures every commit has a unique version number for development and testing.

## Troubleshooting

### "version 0.0.0+unknown"
- **Cause**: No git tags found
- **Fix**: Create a tag: `git tag v0.1.0 && git push origin v0.1.0`

### PyPI upload fails with "File already exists"
- **Cause**: Trying to upload same version twice
- **Fix**: Create a new tag with a higher version number

### Version doesn't match tag
- **Cause**: Not checking out the full git history in CI
- **Fix**: Already configured with `fetch-depth: 0` in workflows

## Files Modified for Automatic Versioning

- **pyproject.toml**: Configured with `dynamic = ["version"]` and `[tool.setuptools_scm]`
- **src/numta/__init__.py**: Imports version from auto-generated `_version.py`
- **.gitignore**: Ignores `src/numta/_version.py` (auto-generated)
- **.github/workflows/**: Uses `fetch-depth: 0` to fetch all git history

## Benefits

✅ **Single source of truth**: Version comes from git tags only
✅ **No version conflicts**: Impossible to forget updating version numbers
✅ **Automatic dev versions**: Every commit has a unique version
✅ **Standard practice**: Used by major Python projects (pytest, numpy, etc.)
