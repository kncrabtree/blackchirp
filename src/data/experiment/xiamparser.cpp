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
    QString groupQuantumNumbers; // Store first 35 characters from group start
    
    while (i < lines.size()) {
        const QString &line = lines[i]; // Don't trim - fixed columns!
        
        // Skip empty lines, comments, rigid lines, and diagnostic output
        if (line.isEmpty() || line.startsWith("#") || line.startsWith("total is") ||
            line.startsWith("Eigenvectors") || line.startsWith("Eigenvalues") ||
            line.contains("Matrix elements") || line.startsWith("\\\\") ||
            line.contains("rigid")) {
            ++i;
            continue;
        }
        
        // Stop at section boundaries
        if (line.contains("---") || line.contains("Maximum") || 
            line.contains("Standard Deviation")) {
            break;
        }
        
        // Check if this starts a new group (first 19 characters trimmed have length > 0)
        QString firstPart = (line.length() >= 19) ? line.left(19).trimmed() : "";
        if (!firstPart.isEmpty()) {
            // Group start - store quantum numbers template
            groupQuantumNumbers = (line.length() >= 35) ? line.left(35) : line;
        }
        
        // Parse the line using group context if needed
        TransitionData transition = parseInts3Line(line, groupQuantumNumbers);
        if (transition.frequency > 0) {
            catalogData.addTransition(transition);
        }
        
        ++i;
    }
    
    return catalogData;
}

double XIAMParser::calculateOptimalIntensity(double linestr, double total, double statWeight, double population, double hvEnergy) const
{
    // Default intensity is total
    if (total <= 0) {
        return 0.0; // No valid intensity available
    }
    
    // Check if linestr, population, and hvEnergy all have 2 or more significant digits
    // statWeight by definition has infinite precision so we use its value as-is
    auto countSignificantDigits = [](double value) -> int {
        if (value <= 0) return 0;
        
        QString str = QString::number(value, 'g', 15);
        str.remove(QRegularExpression("[eE][+-]?\\d+$")); // Remove scientific notation exponent
        str.remove('.');
        str.remove(QRegularExpression("^0+")); // Remove leading zeros
        str.remove(QRegularExpression("0+$")); // Remove trailing zeros
        return str.length();
    };
    
    int linestrDigits = countSignificantDigits(linestr);
    int populationDigits = countSignificantDigits(population);
    int hvEnergyDigits = countSignificantDigits(hvEnergy);
    
    // If linestr, population, and hvEnergy all have 2+ significant digits,
    // return linestr * statWeight * population * hvEnergy
    if (linestrDigits >= 2 && populationDigits >= 2 && hvEnergyDigits >= 2 && 
        linestr > 0 && population > 0 && hvEnergy > 0 && statWeight > 0) {
        return linestr * statWeight * population * hvEnergy;
    }
    
    // Otherwise, return total as the default intensity
    return total;
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
        QString symmetry = parts[6] + " " + parts[7]; // "S" + " " + "1" = "S 1"
        QString vibState = parts[8] + " " + parts[9]; // "V" + " " + "1" = "V 1" 
        
        // Parse frequency and intensity data (now at index 10+) - convert GHz to MHz
        transition.frequency = parts[10].toDouble() * 1000.0;
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



TransitionData XIAMParser::parseInts3Line(const QString &line, const QString &groupQuantumNumbers) const
{
    TransitionData transition;
    
    if (line.length() < 80) {
        return transition; // Line too short to contain full transition data
    }
    
    try {
        QString workingLine = line;
        
        // If this is a split line (no quantum numbers in first 19 chars) and we have group context,
        // construct line with quantum numbers from group
        QString firstPart = (line.length() >= 19) ? line.left(19).trimmed() : "";
        if (firstPart.isEmpty() && !groupQuantumNumbers.isEmpty() && line.length() >= 25) {
            // Split line - inherit quantum numbers but use mode from current line
            workingLine = groupQuantumNumbers;
            if (line.length() >= 25) {
                // Replace characters 23-25 with current line's characters 23-25 (S number)
                QString modeSection = line.mid(20, 16);
                workingLine = workingLine.left(20) + modeSection + 
                             ((line.length() > 36) ? line.mid(36) : "");
            }
        }
        
        QString upperQN, lowerQN, symmetry, vibState, blockNum;
        
        // Parse quantum numbers from working line (positions 0-19)
        QString quantumSection = (workingLine.length() >= 19) ? workingLine.left(19).trimmed() : "";
        QStringList quantumParts = quantumSection.split(QRegularExpression(R"(\s+)"), Qt::SkipEmptyParts);
        
        if (quantumParts.size() >= 6) {
            upperQN = QString("%1 %2 %3").arg(quantumParts[0], quantumParts[1], quantumParts[2]);
            lowerQN = QString("%1 %2 %3").arg(quantumParts[3], quantumParts[4], quantumParts[5]);
        }
        
        // Mode indicators: positions 20-35 (rigid, S #, V #, B #)
        QString modeSection = (workingLine.length() >= 36) ? workingLine.mid(20, 16).trimmed() : "";
        QString mode = "";
        
        if (modeSection.contains("rigid")) {
            mode += "rigid ";
        }
        if (modeSection.contains("S ")) {
            QRegularExpression sRegex(R"(S\s+(\d+))");
            QRegularExpressionMatch sMatch = sRegex.match(modeSection);
            if (sMatch.hasMatch()) {
                symmetry = QString("S %1").arg(sMatch.captured(1));
                mode += symmetry + " ";
            }
        }
        if (modeSection.contains("V ")) {
            QRegularExpression vRegex(R"(V\s+(\d+))");
            QRegularExpressionMatch vMatch = vRegex.match(modeSection);
            if (vMatch.hasMatch()) {
                vibState = QString("V %1").arg(vMatch.captured(1));
                mode += vibState + " ";
            }
        }
        if (modeSection.contains("B ")) {
            QRegularExpression bRegex(R"(B\s+(\d+))");
            QRegularExpressionMatch bMatch = bRegex.match(modeSection);
            if (bMatch.hasMatch()) {
                blockNum = QString("B %1").arg(bMatch.captured(1));
                mode += blockNum + " ";
            }
        }
        transition.additionalData["mode"] = mode.trimmed();
        
        // Frequency: positions 35-47
        QString freqStr = line.mid(35, 13).trimmed();
        if (!freqStr.isEmpty()) {
            double freqGHz = freqStr.toDouble();
            transition.frequency = freqGHz * 1000.0; // Convert GHz to MHz
        }
        
        // Split doesn't matter, ignore it
        
        // Intensity (linestr): positions 57-66 (based on character analysis)
        double linestr = 0.0;
        QString linestrStr = line.mid(57, 9).trimmed();
        if (!linestrStr.isEmpty()) {
            linestr = linestrStr.toDouble();
        }
        
        // Total: positions 67-76 (based on character analysis)
        double total = 0.0;
        if (line.length() > 67+9) {
            QString totalStr = line.mid(67, 9).trimmed();
            if (!totalStr.isEmpty()) {
                total = totalStr.toDouble();
            }
        }
        
        // Statistical weight: positions 76-85
        double statWeight = 0.0;
        if (line.length() > 76+9) {
            QString statStr = line.mid(76, 9).trimmed();
            if (!statStr.isEmpty()) {
                statWeight = statStr.toDouble();
            }
        }
        
        // Population: positions 85-94
        double population = 0.0;
        if (line.length() > 85+9) {
            QString popStr = line.mid(85, 9).trimmed();
            if (!popStr.isEmpty()) {
                population = popStr.toDouble();
            }
        }
        
        // HV-energy: positions 98-109
        double hvEnergy = 0.0;
        if (line.length() > 94+9) {
            QString hvStr = line.mid(94, 9).trimmed();
            if (!hvStr.isEmpty()) {
                hvEnergy = hvStr.toDouble();
            }
        }
        
        // Calculate optimal intensity using the algorithm from the original parser
        transition.intensity = calculateOptimalIntensity(linestr, total, statWeight, population, hvEnergy);
        
        // Set quantum numbers
        if (!upperQN.isEmpty() && !lowerQN.isEmpty()) {
            QString qnString = upperQN + " - " + lowerQN;
            if (!symmetry.isEmpty()) qnString += ", " + symmetry;
            if (!vibState.isEmpty()) qnString += " " + vibState;
            if (!blockNum.isEmpty()) qnString += " " + blockNum;
            transition.quantumNumbers = qnString;
        }
        
        // Store additional data
        transition.additionalData["linestrength"] = linestr;
        transition.additionalData["total"] = total;
        transition.additionalData["statisticalWeight"] = statWeight;
        transition.additionalData["population"] = population;
        transition.additionalData["hvEnergy"] = hvEnergy;
        
        // Parse quantum assignment from the end of the line (remaining part after position 104)
        if (line.length() > 103) {
            QString quantumAssignment = line.mid(103).trimmed();
            if (!quantumAssignment.isEmpty()) {
                transition.additionalData["quantumAssignment"] = quantumAssignment;
            }
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
