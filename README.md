# Refactored_ML

A machine learning project with optimized architecture and clean code principles, following single responsibility and DRY principles.

## Project Structure

```
Refactored_ML/
├── src/               # Source code
├── tests/            # Test files
├── config/           # Configuration files
├── models/           # Trained models (gitignored)
├── data/             # Data files (gitignored)
└── notebooks/        # Jupyter notebooks for exploration
```

## Features

- Clean architecture with separation of concerns
- Configuration management in a single file
- Support for both simulation and live modes using real mainnet data
- No code duplication
- Comprehensive testing suite
- Production-ready implementation

## Setup

1. Clone the repository:
```bash
git clone https://github.com/cetler74/Refactored_ML.git
cd Refactored_ML
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

All configuration options are centralized in `config/config.yaml`. This includes:
- Model parameters
- Training settings
- Environment configurations
- API endpoints and credentials

## Usage

[Usage instructions will be added as the project develops]

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

[License information to be added]

## Contact

[Contact information to be added]