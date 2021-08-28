#ifndef FT_H
#define FT_H

#include <QSharedDataPointer>

#include <QVector>
#include <QPointF>

class FtData;

class Ft
{
public:
    Ft();
    explicit Ft(int numPnts, double f0, double spacing, double loFreqMHz);
    Ft(const Ft &);
    Ft &operator=(const Ft &);
    ~Ft();

    void setPoint(int i, double y, double ignoreRange = 0.0);
    void resize(int n, double ignoreRange = 0.0);
    double &operator[](int i);
    void reserve(int n);
    void squeeze();
    void setLoFreq(double f);
    void setX0(double d);
    void setSpacing(double s);
    void append(double y);
    void trim(double minOffset, double maxOffset);
    void setNumShots(quint64 shots);
    void setData(const QVector<double> d, double yMin, double yMax);

    int size() const;
    bool isEmpty() const;
    double at(int i) const;
    double value(int i) const;
    double constFirst() const;
    double constLast() const;
    double xAt(int i) const;
    double xFirst() const;
    double xLast() const;
    double xSpacing() const;
    double minFreqMHz() const;
    double maxFreqMHz() const;
    double loFreqMHz() const;
    double yMin() const;
    double yMax() const;
    QVector<double> xData() const;
    QVector<double> yData() const;
    QVector<QPointF> toVector() const;
    quint64 shots() const;

private:
    QSharedDataPointer<FtData> data;
};

#endif // FT_H
