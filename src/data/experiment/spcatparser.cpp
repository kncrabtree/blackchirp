#include "spcatparser.h"
#include <QFile>
#include <QTextStream>
#include <QFileInfo>
#include <QRegularExpression>
#include <QDebug>
#include <QtMath>

SPCATParser::SPCATParser()
{
}

bool SPCATParser::canParse(const QString &filePath, const QVariantMap &hints) const
{
    Q_UNUSED(hints)
    
    // Check file extension
    QFileInfo fileInfo(filePath);
    if (!fileExtensions().contains("*." + fileInfo.suffix().toLower())) {
        return false;
    }
    
    // Validate format by examining file structure
    return validateFormat(filePath);
}

CatalogData SPCATParser::parse(const QString &filePath, const QVariantMap &hints) const
{
    Q_UNUSED(hints)
    
    CatalogData catalogData;
    QFile file(filePath);
    
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qWarning() << "SPCATParser: Cannot open file" << filePath;
        return catalogData;
    }
    
    QTextStream stream(&file);
    QVector<TransitionData> transitions;
    int lineNumber = 0;
    int validLines = 0;
    
    // Extract molecule name from filename
    QFileInfo fileInfo(filePath);
    QString moleculeName = fileInfo.baseName();
    
    while (!stream.atEnd()) {
        QString line = stream.readLine();
        lineNumber++;
        
        // Skip empty lines
        if (line.trimmed().isEmpty()) {
            continue;
        }
        
        // Ensure line is exactly 80 characters (pad if necessary)
        if (line.length() < 80) {
            line = line.leftJustified(80);
        } else if (line.length() > 80) {
            line = line.left(80);
        }
        
        TransitionData transition = parseLine(line);
        if (transition.frequency > 0) {  // Valid transition
            transitions.append(transition);
            validLines++;
        }
    }
    
    file.close();
    
    if (transitions.isEmpty()) {
        return catalogData;
    }
    
    // Set up catalog data
    catalogData.setTransitions(transitions);
    catalogData.setSourceProgram("SPCAT");
    catalogData.setMoleculeName(moleculeName);
    catalogData.setMetadataValue("totalLines", lineNumber);
    catalogData.setMetadataValue("validTransitions", validLines);
    catalogData.setMetadataValue("sourceFile", filePath);
        
    return catalogData;
}

QString SPCATParser::formatName() const
{
    return "SPCAT";
}

QString SPCATParser::formatDescription() const
{
    return "SPCAT catalog files (.cat) from the Pickett SPFIT/SPCAT package";
}

QStringList SPCATParser::fileExtensions() const
{
    return {"*.cat"};
}

TransitionData SPCATParser::parseLine(const QString &line) const
{
    TransitionData transition;
    
    // SPCAT fixed-width format:
    // Positions:  1-13: frequency (MHz)
    //            14-21: error (MHz) 
    //            22-29: log intensity (log10(nm²MHz))
    //            30-31: degeneracy
    //            32-41: lower state energy (cm⁻¹)
    //            42-44: upper state degeneracy
    //            45-51: species tag
    //            52-55: format code
    //            56-80: quantum numbers (first half upper, second half lower)
    
    bool ok;
    
    // Parse frequency (positions 1-13)
    QString freqStr = line.mid(0, 13).trimmed();
    transition.frequency = freqStr.toDouble(&ok);
    if (!ok || transition.frequency <= 0) {
        return TransitionData(); // Invalid
    }
    
    // Parse intensity (positions 22-29) - convert from log scale
    QString intensityStr = line.mid(21, 8).trimmed();
    double logIntensity = intensityStr.toDouble(&ok);
    if (ok) {
        transition.intensity = convertIntensity(logIntensity);
    } else {
        transition.intensity = 1.0; // Default if parsing fails
    }
    
    // Parse format code (positions 52-55)
    QString formatStr = line.mid(51, 4).trimmed();
    int formatCode = formatStr.toInt(&ok);
    if (!ok) {
        formatCode = 0;
    }
    
    // Parse quantum numbers (positions 56-80)
    transition.quantumNumbers = parseQuantumNumbers(line, formatCode);
    
    // Store additional SPCAT-specific data
    QString errorStr = line.mid(13, 8).trimmed();
    double error = errorStr.toDouble(&ok);
    if (ok) {
        transition.additionalData.insert("frequencyError", error);
    }
    
    QString degStr = line.mid(29, 2).trimmed();
    int degeneracy = degStr.toInt(&ok);
    if (ok) {
        transition.additionalData.insert("degeneracy", degeneracy);
    }
    
    QString elowStr = line.mid(31, 10).trimmed();
    double lowerEnergy = elowStr.toDouble(&ok);
    if (ok) {
        transition.additionalData.insert("lowerStateEnergy", lowerEnergy);
    }
    
    QString ugupStr = line.mid(41, 3).trimmed();
    int upperDegeneracy = ugupStr.toInt(&ok);
    if (ok) {
        transition.additionalData.insert("upperStateDegeneracy", upperDegeneracy);
    }
    
    QString tagStr = line.mid(44, 7).trimmed();
    if (!tagStr.isEmpty()) {
        transition.additionalData.insert("speciesTag", tagStr);
    }
    
    transition.additionalData.insert("formatCode", formatCode);
    
    return transition;
}

QString SPCATParser::parseQuantumNumbers(const QString &line, int formatCode) const
{
    // Extract quantum numbers from positions 56-80 (25 characters total)
    if (line.length() < 56) {
        return QString();
    }
    
    QString qnString = line.mid(55, 25);
    
    // Determine number of quantum numbers per state from format code
    int nqn = formatCode % 10;
    if (nqn == 0) nqn = 6;  // Default to 6 if not specified
    
    // Calculate character width per quantum number (typically 2 chars each)
    // Total 25 chars / 2 states = 12.5 chars per state
    // For 6 quantum numbers, that's about 2 chars per quantum number
    int qnWidth = 2;
    int stateWidth = nqn * qnWidth;
    
    // Extract upper state quantum numbers (first half)
    QString upperQN = qnString.left(stateWidth).trimmed();
    
    // Extract lower state quantum numbers (second half)  
    QString lowerQN = qnString.mid(stateWidth).trimmed();
    
    // Clean up the quantum number strings
    upperQN.replace(QRegularExpression("\\s+"), " ");
    lowerQN.replace(QRegularExpression("\\s+"), " ");
    
    // Remove semicolons (convert to Blackchirp convention)
    upperQN.replace(';', ',');
    lowerQN.replace(';', ',');
    
    // Format as "upper - lower" transition
    if (upperQN.isEmpty() && lowerQN.isEmpty()) {
        return QString();
    } else if (upperQN.isEmpty()) {
        return lowerQN;
    } else if (lowerQN.isEmpty()) {
        return upperQN;
    } else {
        return upperQN + " - " + lowerQN;
    }
}

double SPCATParser::convertIntensity(double logIntensity) const
{
    // SPCAT stores log10(intensity in nm²MHz)
    // Convert to linear scale
    return qPow(10.0, logIntensity);
}

bool SPCATParser::validateFormat(const QString &filePath) const
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return false;
    }
    
    QTextStream stream(&file);
    int validLines = 0;
    int totalLines = 0;
    
    // Check first few lines for SPCAT format
    while (!stream.atEnd() && totalLines < 10) {
        QString line = stream.readLine();
        totalLines++;
        
        if (line.trimmed().isEmpty()) {
            continue;
        }
        
        // Check if line has reasonable length (should be around 80 chars)
        if (line.length() < 50 || line.length() > 100) {
            continue;
        }
        
        // Check if first field looks like a frequency (numeric, reasonable range)
        bool ok;
        QString freqStr = line.left(13).trimmed();
        double freq = freqStr.toDouble(&ok);
        if (ok && freq > 0 && freq < 1e6) { // Reasonable frequency range (0 - 1 THz)
            validLines++;
        }
    }
    
    file.close();
    
    // Require at least 50% of sampled lines to be valid
    return (totalLines > 0 && (double)validLines / totalLines >= 0.5);
}
