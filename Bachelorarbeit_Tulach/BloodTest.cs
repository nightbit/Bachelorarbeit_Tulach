using System;
using Microsoft.ML.Data;
namespace Bachelorarbeit_Tulach
{
	public class BloodTest
	{
		[LoadColumn(0)]
		public int PatientId;
        [LoadColumn(1)]
        public int SpecimenId;
        [LoadColumn(2)]
        public string? SampleTime;
        [LoadColumn(3)]
        public float BunValue;
        [LoadColumn(4)]
        public float CreatinineValue;
        [LoadColumn(5)]
        public float SodiumValue;
        [LoadColumn(6)]
        public float PotassiumValue;
        [LoadColumn(7)]
        public float ChlorideValue;
        [LoadColumn(8)]
        public float CalciumValue;
        [LoadColumn(9)]
        public float AlbuminValue;
        [LoadColumn(10)]
        public float AlkPhoValue;
        [LoadColumn(11)]
        public float AstValue;
        [LoadColumn(12)]
        public float AltValue;
        [LoadColumn(13)]
        public float BilirubinValue;
        [LoadColumn(14)]
        public float CholesterolValue;
        [LoadColumn(15)]
        public float HdlValue;
        [LoadColumn(16)]
        public float LdlValue;
        [LoadColumn(17)]
        public float TriglyceridesValue;
        [LoadColumn(18)]
        public string? PatientGender;
        [LoadColumn(19)]
        public float PatientAge;
    }
	public class BloodTestPrediciton
	{
		[ColumnName("Score")]
		public float PatientAge;
	}
}