# Security Policy

## Supported Versions

Currently, this is an educational repository and does not have production deployments. All versions are considered supported for learning purposes.

## Reporting Security Vulnerabilities

If you discover a security vulnerability in this repository, please report it responsibly.

### How to Report

1. **Do not** open a public issue
2. Send an email to the repository maintainers
3. Include details about the vulnerability
4. Provide steps to reproduce if applicable
5. Allow time for investigation and response

### What to Include

- Description of the vulnerability
- Impact assessment
- Steps to reproduce
- Proof of concept (if safe)
- Suggested fix (if known)

## Security Best Practices

### Model Security

When working with LLMs, consider:

- **Data Privacy**: Do not fine-tune on sensitive or private data
- **Model Weights**: Keep model checkpoints secure
- **API Keys**: Never commit API keys or credentials
- **Input Validation**: Validate user inputs in production

### Environment Security

- **Dependencies**: Keep dependencies updated
- **Virtual Environments**: Use isolated Python environments
- **Access Control**: Limit access to training infrastructure
- **Audit Logs**: Monitor training and evaluation activities

### Data Security

- **Sanitization**: Clean data of sensitive information
- **Encryption**: Encrypt data at rest and in transit
- **Access Controls**: Limit data access
- **Backup**: Securely backup important data

## Known Security Considerations

### Model Weights

- Model checkpoints can be large and may contain learned patterns
- Do not share fine-tuned models trained on sensitive data
- Review model outputs for potential information leakage

### Training Data

- Training data may influence model behavior
- Be aware of data provenance and licensing
- Consider data privacy and consent

### API Keys and Credentials

- Never commit API keys to the repository
- Use environment variables for sensitive configuration
- Rotate credentials regularly
- Use `.env.example` for configuration templates

## Dependency Security

This repository uses several dependencies. Keep them updated:

```bash
pip install --upgrade pip
pip install --upgrade -r requirements.txt
```

### Security Audits

Run security audits regularly:

```bash
pip install safety
safety check
```

## Disclaimer

This is an independent educational resource for learning LLM fine-tuning and AI concepts. It is not affiliated with, endorsed by, or sponsored by OpenAI, Meta, Hugging Face, or any vendor. The maintainers are not responsible for any security issues that may arise from using this environment in production without proper security hardening.

## Contact

For security-related questions or to report vulnerabilities, please open a security advisory through GitHub's security features or contact the maintainers directly.