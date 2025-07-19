#ifndef GENERICXYDATA_H
#define GENERICXYDATA_H

#include <QString>
#include <QStringList>
#include <QVector>
#include <QPointF>
#include <QtMath>

/**
 * @brief Data structure for generic XY data files
 * 
 * This class holds parsed data from generic XY files (CSV, TSV, space-delimited, etc.)
 * along with metadata about the parsing process and file structure.
 */
class GenericXYData
{
public:
    GenericXYData();
    
    // Data access
    QVector<QPointF> data() const { return d_data; }
    void setData(const QVector<QPointF> &data);
    void addDataPoint(const QPointF &point);
    void addDataPoint(double x, double y) { addDataPoint(QPointF(x, y)); }
    
    // Data bounds (automatically tracked)
    double xMin() const { return d_xMin; }
    double xMax() const { return d_xMax; }
    double yMin() const { return d_yMin; }
    double yMax() const { return d_yMax; }
    
    // Metadata
    QString fileName() const { return d_fileName; }
    void setFileName(const QString &fileName) { d_fileName = fileName; }
    
    QString filePath() const { return d_filePath; }
    void setFilePath(const QString &filePath) { d_filePath = filePath; }
    
    // Column information
    QStringList columnNames() const { return d_columnNames; }
    void setColumnNames(const QStringList &names) { d_columnNames = names; }
    
    int xColumn() const { return d_xColumn; }
    void setXColumn(int column) { d_xColumn = column; }
    
    int yColumn() const { return d_yColumn; }
    void setYColumn(int column) { d_yColumn = column; }
    
    QString xColumnName() const { 
        return (d_xColumn >= 0 && d_xColumn < d_columnNames.size()) ? d_columnNames[d_xColumn] : QString("X"); 
    }
    
    QString yColumnName() const { 
        return (d_yColumn >= 0 && d_yColumn < d_columnNames.size()) ? d_columnNames[d_yColumn] : QString("Y"); 
    }
    
    // Parsing information
    QString delimiter() const { return d_delimiter; }
    void setDelimiter(const QString &delimiter) { d_delimiter = delimiter; }
    
    int headerLines() const { return d_headerLines; }
    void setHeaderLines(int lines) { d_headerLines = lines; }
    
    bool hasColumnHeaders() const { return d_hasColumnHeaders; }
    void setHasColumnHeaders(bool hasHeaders) { d_hasColumnHeaders = hasHeaders; }
    
    // File statistics
    int totalLines() const { return d_totalLines; }
    void setTotalLines(int lines) { d_totalLines = lines; }
    
    int dataLines() const { return d_data.size(); }
    
    // Validation
    bool isEmpty() const { return d_data.isEmpty(); }
    bool isValid() const { return !d_data.isEmpty() && !d_fileName.isEmpty(); }
    
    // Error handling
    QString errorMessage() const { return d_errorMessage; }
    void setErrorMessage(const QString &message) { d_errorMessage = message; }
    bool hasError() const { return !d_errorMessage.isEmpty(); }
    
    // Clear data
    void clear();
    
private:
    void updateBounds(const QPointF &point);
    void resetBounds();
    
    QVector<QPointF> d_data;
    QString d_fileName;
    QString d_filePath;
    QStringList d_columnNames;
    int d_xColumn = 0;
    int d_yColumn = 1;
    QString d_delimiter = ",";
    int d_headerLines = 0;
    bool d_hasColumnHeaders = false;
    int d_totalLines = 0;
    QString d_errorMessage;
    
    // Data bounds
    double d_xMin = qQNaN();
    double d_xMax = qQNaN();
    double d_yMin = qQNaN();
    double d_yMax = qQNaN();
};

#endif // GENERICXYDATA_H