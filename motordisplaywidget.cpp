#include "motordisplaywidget.h"
#include "ui_motordisplaywidget.h"

#include <algorithm>

MotorDisplayWidget::MotorDisplayWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MotorDisplayWidget)
{
    ui->setupUi(this);

    ui->zSlider1->setAxis(BlackChirp::MotorX);
    ui->zSlider2->setAxis(BlackChirp::MotorT);
    ui->xySlider1->setAxis(BlackChirp::MotorZ);
    ui->xySlider2->setAxis(BlackChirp::MotorT);
    ui->timeXSlider->setAxis(BlackChirp::MotorX);
    ui->timeYSlider->setAxis(BlackChirp::MotorY);
    ui->timeZSlider->setAxis(BlackChirp::MotorZ);

    d_sliders << ui->zSlider1 << ui->zSlider2 << ui->xySlider1 << ui->xySlider2
              << ui->timeXSlider << ui->timeYSlider << ui->timeZSlider;

    for(int i=0; i<d_sliders.size();i++)
        connect(d_sliders.at(i),&MotorSliderWidget::valueChanged,this,&MotorDisplayWidget::updatePlots);
}

MotorDisplayWidget::~MotorDisplayWidget()
{
    delete ui;
}

void MotorDisplayWidget::prepareForScan(const MotorScan s)
{
    setEnabled(s.isEnabled());

    if(s.isEnabled())
    {
        d_currentScan = s;

        Q_FOREACH(MotorSliderWidget *w,d_sliders)
            w->setRange(s);

        ui->motorZSpectrogramPlot->prepareForScan(s);
        ui->motorTimePlot->prepareForScan(s);
        ui->motorXYSpectrogramPlot->prepareForScan(s);
    }
}

void MotorDisplayWidget::newMotorData(const MotorScan s)
{
    d_currentScan = s;
    updatePlots();
}

void MotorDisplayWidget::updatePlots()
{
    //prepare slices; update plots
    QVector<double> sliceZPlot = d_currentScan.slice(ui->motorZSpectrogramPlot->leftAxis(),ui->motorZSpectrogramPlot->bottomAxis()
                                         ,ui->zSlider1->axis(),ui->zSlider1->currentIndex(),
                                         ui->zSlider2->axis(),ui->zSlider2->currentIndex());

    ui->motorZSpectrogramPlot->updateData(sliceZPlot,d_currentScan.numPoints(ui->motorZSpectrogramPlot->bottomAxis()));

    QVector<double> sliceXYPlot = d_currentScan.slice(ui->motorXYSpectrogramPlot->leftAxis(),ui->motorXYSpectrogramPlot->bottomAxis()
                                         ,ui->xySlider1->axis(),ui->xySlider1->currentIndex(),
                                         ui->xySlider2->axis(),ui->xySlider2->currentIndex());
    ui->motorXYSpectrogramPlot->updateData(sliceXYPlot,d_currentScan.numPoints(ui->motorXYSpectrogramPlot->bottomAxis()));

    QVector<QPointF> timeTrace = d_currentScan.tTrace(ui->timeXSlider->currentIndex(),ui->timeYSlider->currentIndex(),ui->timeZSlider->currentIndex());
    ui->motorTimePlot->updateData(timeTrace);
}
