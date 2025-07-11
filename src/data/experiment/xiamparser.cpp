#include "xiamparser.h"
#include <QFile>
#include <QTextStream>
#include <QFileInfo>
#include <QRegularExpression>
#include <QDebug>
#include <cmath>

bool XIAMParser::canParse(const QString &filePath) const
{
    // Check file extension
    QFileInfo fileInfo(filePath);
    if (!fileExtensions().contains("*." + fileInfo.suffix().toLower())) {
        return false;
    }
    
    // Check if file is readable
    if (!isFileReadable(filePath)) {
        return false;
    }
    
    // Check for XIAM header patterns
    QStringList header = readFileHeader(filePath, 20);
    
    // Look for XIAM signature patterns
    for (const QString &line : header) {
        // XIAM program identification
        if (line.contains("Internal Rotation Calculation") && 
            line.contains("Holger Hartwig")) {
            return true;
        }
        
        // XIAM data section header
        if (line.contains("-- B") && 
            (line.contains("Freq") || line.contains("Split"))) {
            return true;
        }
        
        // XIAM intensity mode indicator
        if (line.contains("ints") && 
            (line.contains(" 2 ") || line.contains(" 3 "))) {
            return true;
        }
    }
    
    return false;
}

CatalogData XIAMParser::parse(const QString &filePath) const
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qWarning() << "XIAMParser: Cannot open file:" << filePath;
        return CatalogData();
    }
    
    QTextStream stream(&file);
    QStringList lines;
    while (!stream.atEnd()) {
        lines.append(stream.readLine());
    }
    file.close();
    
    if (lines.isEmpty()) {
        qWarning() << "XIAMParser: Empty file:" << filePath;
        return CatalogData();
    }
    
    // Detect intensity mode and data start line
    int intensityMode = detectIntensityMode(lines);
    int dataStartLine = findDataStartLine(lines);
    
    
    if (dataStartLine == -1) {
        qWarning() << "XIAMParser: No data section found in file:" << filePath;
        return CatalogData();
    }
    
    // Parse based on intensity mode
    CatalogData catalogData;
    if (intensityMode == 2) {
        catalogData = parseInts2Format(lines, dataStartLine);
    } else if (intensityMode == 3) {
        catalogData = parseInts3Format(lines, dataStartLine);
    } else {
        qWarning() << "XIAMParser: Unknown intensity mode in file:" << filePath;
        return CatalogData();
    }
    
    // Set metadata
    catalogData.setSourceProgram("XIAM");
    catalogData.setMoleculeName(extractMoleculeName(lines, filePath));
    
    return catalogData;
}

QString XIAMParser::formatName() const
{
    return "XIAM";
}

QString XIAMParser::formatDescription() const
{
    return "XIAM (eXtended Internal Axis Method) spectroscopic catalog output. "
           "Supports both ints=2 (simple) and ints=3 (with splitting analysis) modes.";
}

QStringList XIAMParser::fileExtensions() const
{
    return {"*.xo", "*.out"};
}

int XIAMParser::detectIntensityMode(const QStringList &lines) const
{
    // Look for explicit ints setting
    for (const QString &line : lines) {
        QRegularExpression intsRegex(R"(\bints\s+(\d+))");
        QRegularExpressionMatch match = intsRegex.match(line);
        if (match.hasMatch()) {
            int mode = match.captured(1).toInt();
            if (mode == 2 || mode == 3) {
                return mode;
            }
        }
    }
    
    // Fallback: detect by header format
    for (const QString &line : lines) {
        if (line.contains("-- B") && line.contains("Freq")) {
            if (line.contains("Split")) {
                return 3; // ints=3 has Split column
            } else {
                return 2; // ints=2 has no Split column
            }
        }
    }
    
    return 0; // Unknown
}

int XIAMParser::findDataStartLine(const QStringList &lines) const
{
    for (int i = 0; i < lines.size(); ++i) {
        const QString &line = lines[i];
        if (line.contains("-- B") && 
            (line.contains("Freq") || line.contains("Split"))) {
            return i + 1; // Data starts on next line
        }
    }
    return -1;
}

CatalogData XIAMParser::parseInts2Format(const QStringList &lines, int startLine) const
{
    CatalogData catalogData;
    
    for (int i = startLine; i < lines.size(); ++i) {
        const QString &line = lines[i].trimmed();
        
        // Skip empty lines, comments, and diagnostic output
        if (line.isEmpty() || line.startsWith("#") || line.startsWith("total is") ||
            line.startsWith("Eigenvectors") || line.startsWith("Eigenvalues") ||
            line.contains("Matrix elements") || line.startsWith("\\\\")) {
            continue;
        }
        
        // Stop at section boundaries
        if (line.contains("---") || line.contains("Maximum") || 
            line.contains("Standard Deviation")) {
            break;
        }
        
        TransitionData transition = parseInts2Line(line);
        if (transition.frequency > 0) { // Valid transition
            catalogData.addTransition(transition);
        }
    }
    
    return catalogData;
}

CatalogData XIAMParser::parseInts3Format(const QStringList &lines, int startLine) const
{
    CatalogData catalogData;
    
    int i = startLine;
    while (i < lines.size()) {
        const QString &line = lines[i].trimmed();
        
        // Skip empty lines, comments, and diagnostic output
        if (line.isEmpty() || line.startsWith("#") || line.startsWith("total is") ||
            line.startsWith("Eigenvectors") || line.startsWith("Eigenvalues") ||
            line.contains("Matrix elements") || line.startsWith("\\\\")) {
            ++i;
            continue;
        }
        
        // Stop at section boundaries
        if (line.contains("---") || line.contains("Maximum") || 
            line.contains("Standard Deviation")) {
            break;
        }
        
        // Check if this starts a new transition group or is a standalone transition
        if (isInts3GroupStart(line)) {
            int endIndex;
            QList<TransitionData> groupTransitions = parseInts3Group(lines, i, endIndex);
            
            for (const TransitionData &transition : groupTransitions) {
                if (transition.frequency > 0) {
                    catalogData.addTransition(transition);
                }
            }
            
            i = endIndex + 1;
        } else {
            // Try to parse as standalone transition (might be partial group or isolated line)
            TransitionData transition = parseInts3Line(line);
            if (transition.frequency > 0) {
                QString mode = transition.additionalData.value("mode").toString();
                if (!mode.contains("rigid")) {
                    catalogData.addTransition(transition);
                }
            }
            ++i;
        }
    }
    
    return catalogData;
}

double XIAMParser::calculateOptimalIntensity(double linestr, double total, double statWeight, double population, double hvEnergy) const
{
    // Check if we have all required components for calculation
    if (linestr <= 0 || statWeight <= 0 || population <= 0 || hvEnergy <= 0) {
        return linestr; // Fallback to original linestr
    }
    
    // Calculate expected total from components
    double calculatedTotal = linestr * population * hvEnergy * statWeight;
    
    // Determine number of significant digits in linestr
    QString linestrStr = QString::number(linestr, 'f', 10);
    int decimalPos = linestrStr.indexOf('.');
    int significantDigits = 0;
    bool foundNonZero = false;
    
    for (int i = (decimalPos == -1 ? 0 : decimalPos + 1); i < linestrStr.length(); ++i) {
        if (linestrStr[i].isDigit()) {
            if (linestrStr[i] != '0' || foundNonZero) {
                foundNonZero = true;
                significantDigits++;
            }
        }
    }
    
    // If linestr has very few significant digits (≤3) and calculated total is close to observed total,
    // use the calculated intensity for better precision
    if (significantDigits <= 3 && std::abs(calculatedTotal - total) / total < 0.1) {
        // Calculate more precise intensity: total / (population * hvEnergy * statWeight)
        return total / (population * hvEnergy * statWeight);
    }
    
    // Fallback condition: if intensity is very low (<0.0100) and calculation gives reasonable result
    if (linestr < 0.0100 && calculatedTotal > 0 && std::abs(calculatedTotal - total) / total < 0.2) {
        return total / (population * hvEnergy * statWeight);
    }
    
    return linestr; // Use original value
}

TransitionData XIAMParser::parseInts2Line(const QString &line) const
{
    TransitionData transition;
    
    // XIAM ints=2 format: J K- K+ J K- K+ S V Freq Linestr. total stat.w. popul. hv-ener. [quantum assignment]
    QStringList parts = line.split(QRegularExpression(R"(\s+)"), Qt::SkipEmptyParts);
    
    if (parts.size() < 16) {
        return transition; // Invalid line
    }
    
    try {
        // Extract quantum numbers (first 6 parts)
        QString upperQN = QString("%1 %2 %3").arg(parts[0], parts[1], parts[2]);
        QString lowerQN = QString("%1 %2 %3").arg(parts[3], parts[4], parts[5]);
        QString symmetry = parts[6] + parts[7]; // "S" + "1" = "S1"
        QString vibState = parts[8] + parts[9]; // "V" + "1" = "V1" 
        
        // Parse frequency and intensity data (now at index 10+)
        transition.frequency = parts[10].toDouble();
        double linestr = parts[11].toDouble();
        double total = parts[12].toDouble();
        double statWeight = parts[13].toDouble();
        double population = (parts.size() > 14) ? parts[14].toDouble() : 0.0;
        double hvEnergy = (parts.size() > 15) ? parts[15].toDouble() : 0.0;
        
        // Calculate optimal intensity
        transition.intensity = calculateOptimalIntensity(linestr, total, statWeight, population, hvEnergy);
        
        // Store additional data
        transition.additionalData["linestrength"] = linestr;
        transition.additionalData["total"] = total;
        transition.additionalData["statisticalWeight"] = statWeight;
        transition.additionalData["population"] = population;
        transition.additionalData["hvEnergy"] = hvEnergy;
        
        // Format quantum numbers: upperQN - lowerQN, symmetry vibstate
        QString qnString = QString("%1 - %2, %3 %4").arg(upperQN, lowerQN, symmetry, vibState);
        transition.quantumNumbers = qnString;
        
        // Parse quantum number assignment (remaining parts)
        if (parts.size() > 16) {
            QStringList qnAssignment = parts.mid(16);
            transition.additionalData["quantumAssignment"] = parseQuantumNumbers(qnAssignment.join(" "));
        }
        
    } catch (const std::exception &e) {
        qWarning() << "XIAMParser: Error parsing ints=2 line:" << line;
        return TransitionData(); // Return invalid transition
    }
    
    return transition;
}

QList<TransitionData> XIAMParser::parseInts3Group(const QStringList &lines, int startIndex, int &endIndex) const
{
    QList<TransitionData> transitions;
    double referenceFreq = 0.0;
    
    endIndex = startIndex;
    
    // Parse first line (usually rigid rotor or S 1 reference)
    const QString &firstLine = lines[startIndex].trimmed();
    TransitionData firstTransition = parseInts3Line(firstLine);
    
    if (firstTransition.frequency > 0) {
        referenceFreq = firstTransition.frequency;
        
        // Only add to transitions if it's NOT a rigid rotor reference
        QString mode = firstTransition.additionalData.value("mode").toString();
        if (!mode.contains("rigid")) {
            transitions.append(firstTransition);
        }
    }
    
    // Parse subsequent lines in the group
    int i = startIndex + 1;
    while (i < lines.size()) {
        const QString &line = lines[i].trimmed();
        
        // Stop if we hit an empty line, comment, or new group
        if (line.isEmpty() || line.startsWith("#") || 
            isInts3GroupStart(line) || 
            line.contains("---") || line.contains("Maximum")) {
            break;
        }
        
        // Parse any line that looks like a transition (with or without quantum numbers)
        TransitionData transition;
        
        if (line.startsWith("S ") && !line.contains("V ")) {
            // This is a split line - parse it with reference frequency if available
            transition = parseInts3Line(line, referenceFreq);
        } else if (line.contains("S ") && line.contains("V ")) {
            // This is a full symmetry state line
            transition = parseInts3Line(line);
        } else {
            // Try to parse as a general transition line
            transition = parseInts3Line(line);
        }
        
        // Only add valid transitions (exclude rigid rotor references)
        if (transition.frequency > 0) {
            QString mode = transition.additionalData.value("mode").toString();
            if (!mode.contains("rigid")) {
                transitions.append(transition);
            } else if (referenceFreq == 0.0) {
                // If no reference frequency was set yet, use this rigid line's frequency
                referenceFreq = transition.frequency;
            }
        }
        
        ++i;
    }
    
    endIndex = i - 1;
    return transitions;
}

bool XIAMParser::isInts3GroupStart(const QString &line) const
{
    // A group starts with full quantum numbers (at least 6 numeric values at start)
    QStringList parts = line.split(QRegularExpression(R"(\s+)"), Qt::SkipEmptyParts);
    
    if (parts.size() < 6) {
        return false;
    }
    
    // Check if first 6 parts are integers (quantum numbers)
    for (int i = 0; i < 6; ++i) {
        bool ok;
        parts[i].toInt(&ok);
        if (!ok) {
            return false;
        }
    }
    
    return true;
}

TransitionData XIAMParser::parseInts3Line(const QString &line, double referenceFreq) const
{
    TransitionData transition;
    
    QStringList parts = line.split(QRegularExpression(R"(\s+)"), Qt::SkipEmptyParts);
    
    if (parts.size() < 8) {
        return transition; // Invalid line
    }
    
    try {
        int partIndex = 0;
        QString upperQN, lowerQN, symmetry, vibState, blockNum;
        
        // Check for leading quantum numbers
        if (parts[0].toInt() > 0) {
            // Full quantum number specification
            upperQN = QString("%1 %2 %3").arg(parts[0], parts[1], parts[2]);
            lowerQN = QString("%1 %2 %3").arg(parts[3], parts[4], parts[5]);
            partIndex = 6;
        }
        
        // Parse mode indicators (rigid, S, V, B)
        QString mode = "";
        while (partIndex < parts.size()) {
            if (parts[partIndex] == "rigid") {
                mode += "rigid ";
                partIndex++;
            } else if (parts[partIndex].startsWith("S")) {
                symmetry = parts[partIndex];
                mode += parts[partIndex] + " ";
                partIndex++;
            } else if (parts[partIndex].startsWith("V")) {
                vibState = parts[partIndex];
                mode += parts[partIndex] + " ";
                partIndex++;
            } else if (parts[partIndex].startsWith("B")) {
                blockNum = parts[partIndex];
                mode += parts[partIndex] + " ";
                partIndex++;
            } else {
                break;
            }
        }
        transition.additionalData["mode"] = mode.trimmed();
        
        // Parse frequency
        if (partIndex < parts.size()) {
            transition.frequency = parts[partIndex].toDouble();
            ++partIndex;
        }
        
        // Parse split value (only in ints=3 split lines)
        double splitValue = 0.0;
        if (referenceFreq > 0 && partIndex < parts.size()) {
            bool splitOk;
            splitValue = parts[partIndex].toDouble(&splitOk);
            if (splitOk) {
                transition.additionalData["split"] = splitValue;
                ++partIndex;
            }
        }
        
        // Parse intensity and other values
        double linestr = 0.0, total = 0.0, statWeight = 0.0, population = 0.0, hvEnergy = 0.0;
        
        if (partIndex < parts.size()) {
            linestr = parts[partIndex].toDouble();
            ++partIndex;
        }
        if (partIndex < parts.size()) {
            total = parts[partIndex].toDouble();
            ++partIndex;
        }
        if (partIndex < parts.size()) {
            statWeight = parts[partIndex].toDouble();
            ++partIndex;
        }
        if (partIndex < parts.size()) {
            population = parts[partIndex].toDouble();
            ++partIndex;
        }
        if (partIndex < parts.size()) {
            hvEnergy = parts[partIndex].toDouble();
            ++partIndex;
        }
        
        // Calculate optimal intensity
        transition.intensity = calculateOptimalIntensity(linestr, total, statWeight, population, hvEnergy);
        
        // Store additional data
        transition.additionalData["linestrength"] = linestr;
        transition.additionalData["total"] = total;
        transition.additionalData["statisticalWeight"] = statWeight;
        transition.additionalData["population"] = population;
        transition.additionalData["hvEnergy"] = hvEnergy;
        
        // Format quantum numbers: upperQN - lowerQN, symmetry vibstate blocknum
        if (!upperQN.isEmpty() && !lowerQN.isEmpty()) {
            QString qnString = QString("%1 - %2").arg(upperQN, lowerQN);
            if (!symmetry.isEmpty()) qnString += ", " + symmetry;
            if (!vibState.isEmpty()) qnString += " " + vibState;
            if (!blockNum.isEmpty()) qnString += " " + blockNum;
            transition.quantumNumbers = qnString;
        }
        
        // Parse quantum assignment (remaining parts)
        if (partIndex < parts.size()) {
            QStringList qnAssignment = parts.mid(partIndex);
            transition.additionalData["quantumAssignment"] = parseQuantumNumbers(qnAssignment.join(" "));
        }
        
    } catch (const std::exception &e) {
        qWarning() << "XIAMParser: Error parsing ints=3 line:" << line;
        return TransitionData(); // Return invalid transition
    }
    
    return transition;
}

QString XIAMParser::parseQuantumNumbers(const QString &qnString) const
{
    // XIAM quantum assignments typically in format: "K ±X ±Y t A B"
    // Return as-is for now, but could be reformatted if needed
    return qnString.trimmed();
}

QString XIAMParser::extractMoleculeName(const QStringList &lines, const QString &filePath) const
{
    // Find the line with "nzyk" (start of control parameters)
    for (int i = 0; i < lines.size(); ++i) {
        if (lines[i].contains("nzyk")) {
            // Molecule name is 2 lines before nzyk
            if (i >= 2) {
                QString moleculeName = lines[i - 2].trimmed();
                if (!moleculeName.isEmpty()) {
                    return moleculeName;
                }
            }
            break;
        }
    }
    
    // Fallback to filename
    QFileInfo fileInfo(filePath);
    return fileInfo.baseName();
}