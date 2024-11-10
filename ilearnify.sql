-- MySQL dump 10.13  Distrib 8.0.33, for Win64 (x86_64)
--
-- Host: 67.207.87.64    Database: udemy
-- ------------------------------------------------------
-- Server version	8.0.39-0ubuntu0.20.04.1

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `Answers`
--

DROP TABLE IF EXISTS `Answers`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Answers` (
  `answer_id` int NOT NULL AUTO_INCREMENT,
  `question_id` int DEFAULT NULL,
  `answer_text` text NOT NULL,
  `is_correct` tinyint(1) NOT NULL,
  PRIMARY KEY (`answer_id`),
  KEY `question_id` (`question_id`),
  CONSTRAINT `Answers_ibfk_1` FOREIGN KEY (`question_id`) REFERENCES `Questions` (`question_id`)
) ENGINE=InnoDB AUTO_INCREMENT=109 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Answers`
--

LOCK TABLES `Answers` WRITE;
/*!40000 ALTER TABLE `Answers` DISABLE KEYS */;
INSERT INTO `Answers` VALUES (17,5,'La programación es el proceso de crear software.',1),(18,5,'Es el diseño de hardware.',0),(19,5,'Es la configuración de redes.',0),(20,5,'Es el uso de aplicaciones.',0),(21,6,'JavaScript',1),(22,6,'COBOL',0),(23,6,'Haskell',0),(24,6,'FORTRAN',0),(25,25,'La programación es el proceso de crear software.',1),(26,25,'Es el diseño de hardware.',0),(27,25,'Es la configuración de redes.',0),(28,25,'Es el uso de aplicaciones.',0),(29,26,'JavaScript',1),(30,26,'COBOL',0),(31,26,'Haskell',0),(32,26,'FORTRAN',0),(33,27,'al que va pasar',0),(34,27,'algo que paso',0),(35,27,'algo que probablemente va pasar',1),(36,27,'nose',0),(37,28,'1',0),(38,28,'2',1),(39,28,'3',0),(40,28,'4',0),(41,29,'8',0),(42,29,'10',0),(43,29,'15',1),(44,29,'18',0),(45,30,'8x+4',1),(46,30,'8x+1',0),(47,30,'6 ? + 4',0),(48,30,'8x+2',0),(49,31,'Propiedad Asociativa',0),(50,31,'Propiedad Distributiva',0),(51,31,'Propiedad Conmutativa',1),(52,31,'Propiedad Identidad',0),(53,32,'1',0),(54,32,'2',1),(55,32,'3',0),(56,32,'4',0),(57,33,'2',0),(58,33,'3',0),(59,33,'4',1),(60,33,'5',0),(61,34,'x<4',1),(62,34,'x<3',0),(63,34,'? < 5',0),(64,34,'x<6',0),(65,35,'1',0),(66,35,'2',1),(67,35,'3',0),(68,35,'4',0),(69,36,'',0),(70,36,'',1),(71,36,'',0),(72,36,'',0),(73,37,'2 canicas',0),(74,37,'4 canicas',0),(75,37,'8 canicas',1),(76,37,'6 canicas',0),(77,38,'1/3 de pan chuta.',0),(78,38,'3/4 de pan chuta',0),(79,38,'1*3/4 de pan chuta.',1),(80,38,'1*1/3 de pan chuta.',0),(81,39,'8 panes.',0),(82,39,'6 panes.',1),(83,39,'4 panes.',0),(84,39,'2 panes.',0),(85,40,'15 bolsas.',0),(86,40,'20 bolsas.',1),(87,40,'31 bolsas',0),(88,40,'32 bolsas',0),(89,41,'16 mujeres',0),(90,41,'24 mujeres. ',0),(91,41,'36 mujeres. ',1),(92,41,'44 mujeres.',0),(93,42,'25',0),(94,42,'23',1),(95,42,'8',0),(96,42,'6',0),(97,43,'',0),(98,43,'',0),(99,43,'',0),(100,43,'',0),(101,44,'1',0),(102,44,'2',1),(103,44,'3',0),(104,44,'4',0),(105,45,'1',0),(106,45,'2',0),(107,45,'3',1),(108,45,'4',0);
/*!40000 ALTER TABLE `Answers` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Courses`
--

DROP TABLE IF EXISTS `Courses`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Courses` (
  `course_id` int NOT NULL AUTO_INCREMENT,
  `title` varchar(100) NOT NULL,
  `description` text NOT NULL,
  `teacher_id` int DEFAULT NULL,
  `grade_id` int DEFAULT NULL,
  PRIMARY KEY (`course_id`),
  KEY `teacher_id` (`teacher_id`),
  KEY `grade_id` (`grade_id`),
  CONSTRAINT `Courses_ibfk_1` FOREIGN KEY (`teacher_id`) REFERENCES `Teachers` (`teacher_id`),
  CONSTRAINT `Courses_ibfk_2` FOREIGN KEY (`grade_id`) REFERENCES `Grades` (`grade_id`)
) ENGINE=InnoDB AUTO_INCREMENT=49 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Courses`
--

LOCK TABLES `Courses` WRITE;
/*!40000 ALTER TABLE `Courses` DISABLE KEYS */;
INSERT INTO `Courses` VALUES (5,'Arte y Cultura - Grado 2','El curso de Arte inserta al alumno en el mundo de la cultura a través del conocimiento y apreciación de las manifestaciones artísticas y culturales de las diferentes épocas.',1,NULL),(6,'Razonamiento matematico','En este curso razonaras',1,NULL),(7,'Razonamiento verbal','En este curso razonaras',1,NULL),(8,'Razonamiento','En este curso razonaras',1,NULL),(9,'Comunicación','En este curso aprenderas...',1,NULL),(13,'Inglés Grado 2','Gramática básica, vocabulario y comprensión de lectura.',1,NULL),(14,'Historia Grado 2','En curso de Historia del Perú',1,NULL),(15,'Aritmetica','Curso Aritmetica',1,NULL),(16,'Algebra','El álgebra es la rama de la matemática que estudia la combinación de elementos como números, letras y signos para elaborar diferentes operaciones aritméticas elementales.',1,1),(17,'Matemáticas Grado 1','Introducción al álgebra básica y geometría.',1,1),(18,'Inglés Grado 1','Gramática básica, vocabulario y comprensión de lectura.',1,1),(19,'Ciencias Grado 1','Introducción a biología, química y física.',1,1),(20,'Historia Grado 1','Visión general de la historia local y mundial.',1,1),(21,'Educación Física Grado 1','Introducción a deportes y condición física.',1,1),(22,'Arte Grado 1','Exploración de diversas técnicas y estilos artísticos.',1,1),(23,'Matemáticas Grado 2','Álgebra y geometría avanzada.',1,2),(24,'Inglés Grado 2','Gramática, composición y análisis literario.',1,2),(25,'Ciencias Grado 2','Estudio profundo de biología, química y física.',1,2),(26,'Historia Grado 2','Examen detallado de eventos históricos y figuras.',1,2),(27,'Educación Física Grado 2','Habilidades deportivas avanzadas y entrenamiento físico.',1,2),(28,'Arte Grado 2','Exploración de técnicas artísticas avanzadas y creatividad.',1,2),(29,'Matemáticas Grado 3','Preparación para álgebra y geometría de nivel superior.',1,3),(30,'Inglés Grado 3','Gramática avanzada, habilidades de escritura y literatura.',1,3),(31,'Ciencias Grado 3','Exploración profunda de principios científicos y experimentos.',1,3),(32,'Historia Grado 3','Estudio integral de la historia local y mundial.',1,3),(33,'Educación Física Grado 3','Competiciones deportivas y desafíos de condición física.',1,3),(34,'Arte Grado 3','Proyectos avanzados en diversas formas y estilos artísticos.',1,3),(35,'Matemáticas Grado 4','Álgebra y cálculo avanzado.',1,4),(36,'Inglés Grado 4','Análisis literario y habilidades avanzadas de escritura.',1,4),(37,'Ciencias Grado 4','Estudio avanzado en biología, química y física.',1,4),(38,'Historia Grado 4','Análisis profundo de eventos históricos y movimientos culturales.',1,4),(39,'Educación Física Grado 4','Entrenamiento físico y psicología deportiva.',1,4),(40,'Arte Grado 4','Expresión artística avanzada y desarrollo de portafolios.',1,4),(41,'Matemáticas Grado 5','Preparación para cálculo avanzado y modelado matemático.',1,5),(42,'Inglés Grado 5','Análisis crítico de literatura y técnicas avanzadas de escritura.',1,5),(43,'Ciencias Grado 5','Investigación avanzada y experimentos en biología, química y física.',1,5),(44,'Historia Grado 5','Perspectivas globales sobre eventos históricos y movimientos sociales.',1,5),(45,'Educación Física Grado 5','Entrenamiento deportivo avanzado y nutrición.',1,5),(46,'Arte Grado 5','Exploración de movimientos artísticos contemporáneos y técnicas.',1,5),(48,'Curso 1','prueba curso 1',1,2);
/*!40000 ALTER TABLE `Courses` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Enrollments`
--

DROP TABLE IF EXISTS `Enrollments`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Enrollments` (
  `enrollment_id` int NOT NULL AUTO_INCREMENT,
  `student_id` int DEFAULT NULL,
  `course_id` int DEFAULT NULL,
  `enrollment_date` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`enrollment_id`),
  KEY `student_id` (`student_id`),
  KEY `course_id` (`course_id`),
  CONSTRAINT `Enrollments_ibfk_1` FOREIGN KEY (`student_id`) REFERENCES `Students` (`student_id`),
  CONSTRAINT `Enrollments_ibfk_2` FOREIGN KEY (`course_id`) REFERENCES `Courses` (`course_id`)
) ENGINE=InnoDB AUTO_INCREMENT=24 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Enrollments`
--

LOCK TABLES `Enrollments` WRITE;
/*!40000 ALTER TABLE `Enrollments` DISABLE KEYS */;
INSERT INTO `Enrollments` VALUES (20,158,16,'2024-06-17 00:21:06'),(21,230,23,'2024-06-17 23:08:28'),(22,230,25,'2024-06-18 15:21:06'),(23,230,28,'2024-06-18 15:21:16');
/*!40000 ALTER TABLE `Enrollments` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `ExamResults`
--

DROP TABLE IF EXISTS `ExamResults`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `ExamResults` (
  `result_id` int NOT NULL AUTO_INCREMENT,
  `student_id` int DEFAULT NULL,
  `exam_id` int DEFAULT NULL,
  `correct_answers` int DEFAULT NULL,
  `wrong_answers` int DEFAULT NULL,
  `total_score` decimal(5,2) DEFAULT NULL,
  `result_date` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`result_id`),
  KEY `student_id` (`student_id`),
  KEY `exam_id` (`exam_id`),
  CONSTRAINT `ExamResults_ibfk_1` FOREIGN KEY (`student_id`) REFERENCES `Students` (`student_id`),
  CONSTRAINT `ExamResults_ibfk_2` FOREIGN KEY (`exam_id`) REFERENCES `Exams` (`exam_id`)
) ENGINE=InnoDB AUTO_INCREMENT=218 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ExamResults`
--

LOCK TABLES `ExamResults` WRITE;
/*!40000 ALTER TABLE `ExamResults` DISABLE KEYS */;
INSERT INTO `ExamResults` VALUES (139,158,26,20,5,19.58,'2024-06-15 04:51:54'),(140,159,26,18,7,15.79,'2024-06-15 04:51:54'),(141,160,26,19,6,15.20,'2024-06-15 04:51:54'),(142,161,26,17,8,18.62,'2024-06-15 04:51:54'),(143,162,26,16,9,17.50,'2024-06-15 04:51:54'),(144,163,27,22,3,16.66,'2024-06-15 04:51:54'),(145,164,27,20,5,15.80,'2024-06-15 04:51:54'),(146,165,27,18,7,19.02,'2024-06-15 04:51:54'),(147,166,27,21,4,17.68,'2024-06-15 04:51:54'),(148,167,27,19,6,16.33,'2024-06-15 04:51:54'),(149,168,28,19,6,18.62,'2024-06-15 04:51:54'),(150,169,28,17,8,19.13,'2024-06-15 04:51:54'),(151,170,28,20,5,19.78,'2024-06-15 04:51:54'),(152,171,28,18,7,16.51,'2024-06-15 04:51:54'),(153,172,28,16,9,18.23,'2024-06-15 04:51:54'),(154,173,29,21,4,16.61,'2024-06-15 04:51:54'),(155,174,29,19,6,18.34,'2024-06-15 04:51:54'),(156,175,29,18,7,16.89,'2024-06-15 04:51:54'),(157,176,29,20,5,19.42,'2024-06-15 04:51:54'),(158,177,29,17,8,16.44,'2024-06-15 04:51:54'),(159,178,30,20,5,18.93,'2024-06-15 04:51:54'),(160,179,30,22,3,15.33,'2024-06-15 04:51:54'),(161,180,30,21,4,19.87,'2024-06-15 04:51:54'),(162,181,30,19,6,18.35,'2024-06-15 04:51:54'),(163,182,30,18,7,17.15,'2024-06-15 04:51:54'),(164,183,31,18,7,15.72,'2024-06-15 04:51:54'),(165,184,31,19,6,17.11,'2024-06-15 04:51:54'),(166,185,31,21,4,18.42,'2024-06-15 04:51:54'),(167,186,31,20,5,15.77,'2024-06-15 04:51:54'),(168,187,31,17,8,18.57,'2024-06-15 04:51:54'),(169,188,32,22,3,15.54,'2024-06-15 04:51:54'),(170,189,32,20,5,17.00,'2024-06-15 04:51:54'),(171,190,32,18,7,18.39,'2024-06-15 04:51:54'),(172,191,32,21,4,15.96,'2024-06-15 04:51:54'),(173,192,32,19,6,19.60,'2024-06-15 04:51:54'),(174,193,33,20,5,15.12,'2024-06-15 04:51:54'),(175,194,33,18,7,16.80,'2024-06-15 04:51:54'),(176,195,33,22,3,18.65,'2024-06-15 04:51:54'),(177,196,33,19,6,17.86,'2024-06-15 04:51:54'),(178,197,33,21,4,18.33,'2024-06-15 04:51:54'),(179,198,34,19,6,18.06,'2024-06-15 04:51:54'),(180,199,34,21,4,15.31,'2024-06-15 04:51:54'),(181,200,34,20,5,17.39,'2024-06-15 04:51:54'),(182,201,34,18,7,16.01,'2024-06-15 04:51:54'),(183,202,34,17,8,17.89,'2024-06-15 04:51:54'),(184,203,35,21,4,16.39,'2024-06-15 04:51:54'),(185,204,35,19,6,18.31,'2024-06-15 04:51:54'),(186,205,35,20,5,17.38,'2024-06-15 04:51:54'),(187,206,35,18,7,16.95,'2024-06-15 04:51:54'),(188,207,35,22,3,17.62,'2024-06-15 04:51:54'),(189,208,36,20,5,17.23,'2024-06-15 04:51:54'),(190,209,36,19,6,18.31,'2024-06-15 04:51:54'),(191,210,36,21,4,19.85,'2024-06-15 04:51:54'),(192,211,36,18,7,19.32,'2024-06-15 04:51:54'),(193,212,36,22,3,17.04,'2024-06-15 04:51:54'),(194,213,37,22,3,17.27,'2024-06-15 04:51:54'),(195,214,37,21,4,15.20,'2024-06-15 04:51:54'),(196,215,37,20,5,19.20,'2024-06-15 04:51:54'),(197,216,37,18,7,15.38,'2024-06-15 04:51:54'),(198,217,37,19,6,19.33,'2024-06-15 04:51:54'),(199,218,38,21,4,15.48,'2024-06-15 04:51:54'),(200,219,38,20,5,19.42,'2024-06-15 04:51:54'),(201,220,38,22,3,15.67,'2024-06-15 04:51:54'),(202,221,38,19,6,15.07,'2024-06-15 04:51:54'),(203,222,38,18,7,18.33,'2024-06-15 04:51:54'),(204,223,39,20,5,16.46,'2024-06-15 04:51:54'),(205,224,39,21,4,17.32,'2024-06-15 04:51:54'),(206,225,39,19,6,17.19,'2024-06-15 04:51:54'),(207,226,39,22,3,19.01,'2024-06-15 04:51:54'),(208,227,39,18,7,18.47,'2024-06-15 04:51:54'),(209,228,40,21,4,15.32,'2024-06-15 04:51:54'),(210,229,40,20,5,16.21,'2024-06-15 04:51:54'),(211,158,22,3,0,20.00,'2024-06-15 23:39:01'),(212,158,23,3,2,12.00,'2024-06-15 23:40:48'),(213,158,29,0,0,0.00,'2024-06-15 23:41:09'),(214,158,24,1,2,6.67,'2024-06-15 23:41:22'),(215,158,25,0,1,0.00,'2024-06-15 23:41:28'),(216,158,31,0,0,0.00,'2024-06-15 23:41:33'),(217,230,56,3,3,10.00,'2024-06-17 23:24:58');
/*!40000 ALTER TABLE `ExamResults` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Exams`
--

DROP TABLE IF EXISTS `Exams`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Exams` (
  `exam_id` int NOT NULL AUTO_INCREMENT,
  `unit_id` int DEFAULT NULL,
  `title` varchar(100) NOT NULL,
  `course_id` int DEFAULT NULL,
  `create_date` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`exam_id`),
  KEY `unit_id` (`unit_id`),
  KEY `course_id` (`course_id`),
  CONSTRAINT `course_id` FOREIGN KEY (`course_id`) REFERENCES `Courses` (`course_id`),
  CONSTRAINT `Exams_ibfk_1` FOREIGN KEY (`unit_id`) REFERENCES `Units` (`unit_id`)
) ENGINE=InnoDB AUTO_INCREMENT=59 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Exams`
--

LOCK TABLES `Exams` WRITE;
/*!40000 ALTER TABLE `Exams` DISABLE KEYS */;
INSERT INTO `Exams` VALUES (3,2,'Examen de prueba',6,'2024-06-12 23:11:41'),(22,5,'Examen de prediccicon',14,'2024-06-12 23:11:41'),(23,11,'Examen Unidad 1',16,'2024-06-12 23:11:41'),(24,12,'Examen Unidad 2',16,'2024-06-12 23:11:41'),(25,13,'EXAMEN 1',5,'2024-06-13 03:16:07'),(26,19,'Examen Unidad 1 - Álgebra básica',16,'2024-06-15 04:45:46'),(27,20,'Examen Unidad 2 - Geometría elemental',16,'2024-06-15 04:45:46'),(28,21,'Examen Unidad 3 - Problemas de aplicación',16,'2024-06-15 04:45:46'),(29,22,'Examen Unidad 4 - Ecuaciones lineales',16,'2024-06-15 04:45:46'),(30,23,'Examen Unidad 5 - Geometría avanzada',16,'2024-06-15 04:45:46'),(31,24,'Examen Unidad 1 - Gramática básica',17,'2024-06-15 04:45:46'),(32,25,'Examen Unidad 2 - Vocabulario esencial',17,'2024-06-15 04:45:46'),(33,26,'Examen Unidad 3 - Comprensión auditiva',17,'2024-06-15 04:45:46'),(34,27,'Examen Unidad 4 - Comprensión de lectura',17,'2024-06-15 04:45:46'),(35,28,'Examen Unidad 5 - Conversación básica',17,'2024-06-15 04:45:46'),(36,29,'Examen Unidad 1 - Introducción a la biología',18,'2024-06-15 04:45:46'),(37,30,'Examen Unidad 2 - Fundamentos de química',18,'2024-06-15 04:45:46'),(38,31,'Examen Unidad 3 - Conceptos básicos de física',18,'2024-06-15 04:45:46'),(39,32,'Examen Unidad 4 - Experimentación científica',18,'2024-06-15 04:45:46'),(40,33,'Examen Unidad 5 - Teoría atómica y molecular',18,'2024-06-15 04:45:46'),(41,34,'Examen Unidad 1 - Historia local',19,'2024-06-15 04:45:47'),(42,35,'Examen Unidad 2 - Culturas antiguas',19,'2024-06-15 04:45:47'),(43,36,'Examen Unidad 3 - Edad Media y Renacimiento',19,'2024-06-15 04:45:47'),(44,37,'Examen Unidad 4 - Revoluciones e independencias',19,'2024-06-15 04:45:47'),(45,38,'Examen Unidad 5 - Siglo XX: Guerras y globalización',19,'2024-06-15 04:45:47'),(46,39,'Examen Unidad 1 - Introducción al deporte',20,'2024-06-15 04:45:47'),(47,40,'Examen Unidad 2 - Fundamentos del ejercicio físico',20,'2024-06-15 04:45:47'),(48,41,'Examen Unidad 3 - Técnicas deportivas',20,'2024-06-15 04:45:47'),(49,42,'Examen Unidad 4 - Salud y bienestar',20,'2024-06-15 04:45:47'),(50,43,'Examen Unidad 5 - Competencias y habilidades físicas',20,'2024-06-15 04:45:47'),(51,44,'Examen Unidad 1 - Introducción al arte',21,'2024-06-15 04:45:47'),(52,45,'Examen Unidad 2 - Técnicas artísticas básicas',21,'2024-06-15 04:45:47'),(53,46,'Examen Unidad 3 - Estilos y corrientes artísticas',21,'2024-06-15 04:45:47'),(54,47,'Examen Unidad 4 - Expresión creativa',21,'2024-06-15 04:45:47'),(55,48,'Examen Unidad 5 - Proyecto final: Creación artística',21,'2024-06-15 04:45:47'),(56,49,'Examen 1',23,'2024-06-17 23:23:05'),(57,2,'',6,'2024-06-18 04:39:31'),(58,7,'EXAMEN 1',5,'2024-06-18 15:36:39');
/*!40000 ALTER TABLE `Exams` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Grades`
--

DROP TABLE IF EXISTS `Grades`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Grades` (
  `grade_id` int NOT NULL AUTO_INCREMENT,
  `grade_name` enum('1','2','3','4','5') NOT NULL,
  PRIMARY KEY (`grade_id`)
) ENGINE=InnoDB AUTO_INCREMENT=7 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Grades`
--

LOCK TABLES `Grades` WRITE;
/*!40000 ALTER TABLE `Grades` DISABLE KEYS */;
INSERT INTO `Grades` VALUES (1,'1'),(2,'2'),(3,'3'),(4,'4'),(5,'5');
/*!40000 ALTER TABLE `Grades` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Library`
--

DROP TABLE IF EXISTS `Library`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Library` (
  `library_id` int NOT NULL AUTO_INCREMENT,
  `course_id` int DEFAULT NULL,
  `document_title` varchar(255) DEFAULT NULL,
  `document_description` text,
  `document_path` varchar(255) DEFAULT NULL,
  `extension` varchar(10) DEFAULT NULL,
  PRIMARY KEY (`library_id`),
  KEY `course_id` (`course_id`),
  CONSTRAINT `Library_ibfk_1` FOREIGN KEY (`course_id`) REFERENCES `Courses` (`course_id`)
) ENGINE=InnoDB AUTO_INCREMENT=15 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Library`
--

LOCK TABLES `Library` WRITE;
/*!40000 ALTER TABLE `Library` DISABLE KEYS */;
INSERT INTO `Library` VALUES (13,9,'Comunicación 1','Libro digital - Comunicación 1','\\var\\www\\images\\document-1718653145827-816454903.pdf','pdf'),(14,25,'Ciencia, Tecnología y Ambiente 2 ','Libro digital ','\\var\\www\\images\\document-1718663123113-601979125.pdf','pdf');
/*!40000 ALTER TABLE `Library` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Materials`
--

DROP TABLE IF EXISTS `Materials`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Materials` (
  `material_id` int NOT NULL AUTO_INCREMENT,
  `unit_id` int DEFAULT NULL,
  `course_id` int DEFAULT NULL,
  `title` varchar(100) DEFAULT NULL,
  `description` text,
  `path` varchar(255) DEFAULT NULL,
  `extension` varchar(10) DEFAULT NULL,
  PRIMARY KEY (`material_id`),
  KEY `unit_id` (`unit_id`),
  KEY `course_id` (`course_id`),
  CONSTRAINT `Materials_ibfk_1` FOREIGN KEY (`unit_id`) REFERENCES `Units` (`unit_id`),
  CONSTRAINT `Materials_ibfk_2` FOREIGN KEY (`course_id`) REFERENCES `Courses` (`course_id`)
) ENGINE=InnoDB AUTO_INCREMENT=10 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Materials`
--

LOCK TABLES `Materials` WRITE;
/*!40000 ALTER TABLE `Materials` DISABLE KEYS */;
INSERT INTO `Materials` VALUES (9,49,23,'Introducción a las matematicas ','Cuaderno de trabajo ','\\var\\www\\images\\1496bd6d02c1a7c420b9a5ac3d6a6d16.pdf','pdf');
/*!40000 ALTER TABLE `Materials` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Prediction`
--

DROP TABLE IF EXISTS `Prediction`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Prediction` (
  `idPrediction` int NOT NULL AUTO_INCREMENT,
  `studentId` int DEFAULT NULL,
  `courseId` int DEFAULT NULL,
  `value` double DEFAULT NULL,
  `description` varchar(45) DEFAULT NULL,
  PRIMARY KEY (`idPrediction`)
) ENGINE=InnoDB AUTO_INCREMENT=256 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Prediction`
--

LOCK TABLES `Prediction` WRITE;
/*!40000 ALTER TABLE `Prediction` DISABLE KEYS */;
INSERT INTO `Prediction` VALUES (41,161,16,0.182,'prediccion de bueno'),(42,161,16,0.273,'prediccion de malo'),(43,161,16,0.182,'prediccion de muy bueno'),(44,161,16,0.273,'prediccion de no pase'),(45,161,16,0.091,'prediccion de regularmente'),(46,159,16,0.091,'prediccion de bueno'),(47,159,16,0.273,'prediccion de malo'),(48,159,16,0.182,'prediccion de muy bueno'),(49,159,16,0.364,'prediccion de no pase'),(50,159,16,0.091,'prediccion de regularmente'),(51,163,16,0.182,'prediccion de bueno'),(52,163,16,0.182,'prediccion de malo'),(53,163,16,0.273,'prediccion de muy bueno'),(54,163,16,0.273,'prediccion de no pase'),(55,163,16,0.091,'prediccion de regularmente'),(56,162,16,0.182,'prediccion de bueno'),(57,162,16,0.091,'prediccion de malo'),(58,162,16,0.273,'prediccion de muy bueno'),(59,162,16,0.273,'prediccion de no pase'),(60,162,16,0.182,'prediccion de regularmente'),(61,158,5,0.257,'prediccion de bueno'),(62,158,5,0.201,'prediccion de malo'),(63,158,5,0.174,'prediccion de muy bueno'),(64,158,5,0.101,'prediccion de no pase'),(65,158,5,0.268,'prediccion de regularmente'),(66,158,14,0.286,'prediccion de bueno'),(67,158,14,0.226,'prediccion de malo'),(68,158,14,0.173,'prediccion de muy bueno'),(69,158,14,0.194,'prediccion de no pase'),(70,158,14,0.122,'prediccion de regularmente'),(71,158,16,0.259,'prediccion de bueno'),(72,158,16,0.219,'prediccion de malo'),(73,158,16,0.17,'prediccion de muy bueno'),(74,158,16,0.244,'prediccion de no pase'),(75,158,16,0.108,'prediccion de regularmente'),(76,158,17,0.218,'prediccion de bueno'),(77,158,17,0.203,'prediccion de malo'),(78,158,17,0.165,'prediccion de muy bueno'),(79,158,17,0.304,'prediccion de no pase'),(80,158,17,0.109,'prediccion de regularmente'),(81,164,16,0.091,'prediccion de bueno'),(82,164,16,0.091,'prediccion de malo'),(83,164,16,0.364,'prediccion de muy bueno'),(84,164,16,0.182,'prediccion de no pase'),(85,164,16,0.273,'prediccion de regularmente'),(86,166,16,0.091,'prediccion de bueno'),(87,166,16,0.182,'prediccion de malo'),(88,166,16,0.455,'prediccion de muy bueno'),(89,166,16,0.182,'prediccion de no pase'),(90,166,16,0.091,'prediccion de regularmente'),(91,165,16,0.182,'prediccion de bueno'),(92,165,16,0.182,'prediccion de malo'),(93,165,16,0.182,'prediccion de muy bueno'),(94,165,16,0.364,'prediccion de no pase'),(95,165,16,0.091,'prediccion de regularmente'),(96,171,16,0.091,'prediccion de bueno'),(97,171,16,0.182,'prediccion de malo'),(98,171,16,0.182,'prediccion de muy bueno'),(99,171,16,0.182,'prediccion de no pase'),(100,171,16,0.364,'prediccion de regularmente'),(101,168,16,0.182,'prediccion de bueno'),(102,168,16,0.091,'prediccion de malo'),(103,168,16,0.091,'prediccion de muy bueno'),(104,168,16,0.455,'prediccion de no pase'),(105,168,16,0.182,'prediccion de regularmente'),(106,170,16,0.364,'prediccion de bueno'),(107,170,16,0.091,'prediccion de malo'),(108,170,16,0.091,'prediccion de muy bueno'),(109,170,16,0.364,'prediccion de no pase'),(110,170,16,0.091,'prediccion de regularmente'),(111,172,16,0.091,'prediccion de bueno'),(112,172,16,0.364,'prediccion de malo'),(113,172,16,0.091,'prediccion de muy bueno'),(114,172,16,0.182,'prediccion de no pase'),(115,172,16,0.273,'prediccion de regularmente'),(116,177,16,0.182,'prediccion de bueno'),(117,177,16,0.455,'prediccion de malo'),(118,177,16,0.182,'prediccion de muy bueno'),(119,177,16,0.091,'prediccion de no pase'),(120,177,16,0.091,'prediccion de regularmente'),(121,179,16,0.091,'prediccion de bueno'),(122,179,16,0.091,'prediccion de malo'),(123,179,16,0.273,'prediccion de muy bueno'),(124,179,16,0.364,'prediccion de no pase'),(125,179,16,0.182,'prediccion de regularmente'),(126,183,17,0.273,'prediccion de bueno'),(127,183,17,0.364,'prediccion de malo'),(128,183,17,0.091,'prediccion de muy bueno'),(129,183,17,0.182,'prediccion de no pase'),(130,183,17,0.091,'prediccion de regularmente'),(131,185,17,0.364,'prediccion de bueno'),(132,185,17,0.182,'prediccion de malo'),(133,185,17,0.182,'prediccion de muy bueno'),(134,185,17,0.182,'prediccion de no pase'),(135,185,17,0.091,'prediccion de regularmente'),(136,187,17,0.182,'prediccion de bueno'),(137,187,17,0.273,'prediccion de malo'),(138,187,17,0.182,'prediccion de muy bueno'),(139,187,17,0.091,'prediccion de no pase'),(140,187,17,0.273,'prediccion de regularmente'),(141,191,17,0.091,'prediccion de bueno'),(142,191,17,0.273,'prediccion de malo'),(143,191,17,0.091,'prediccion de muy bueno'),(144,191,17,0.091,'prediccion de no pase'),(145,191,17,0.455,'prediccion de regularmente'),(146,189,17,0.273,'prediccion de bueno'),(147,189,17,0.182,'prediccion de malo'),(148,189,17,0.182,'prediccion de muy bueno'),(149,189,17,0.273,'prediccion de no pase'),(150,189,17,0.091,'prediccion de regularmente'),(151,192,17,0.091,'prediccion de bueno'),(152,192,17,0.182,'prediccion de malo'),(153,192,17,0.364,'prediccion de muy bueno'),(154,192,17,0.091,'prediccion de no pase'),(155,192,17,0.273,'prediccion de regularmente'),(156,188,17,0.091,'prediccion de bueno'),(157,188,17,0.091,'prediccion de malo'),(158,188,17,0.273,'prediccion de muy bueno'),(159,188,17,0.091,'prediccion de no pase'),(160,188,17,0.455,'prediccion de regularmente'),(161,194,17,0.182,'prediccion de bueno'),(162,194,17,0.182,'prediccion de malo'),(163,194,17,0.273,'prediccion de muy bueno'),(164,194,17,0.091,'prediccion de no pase'),(165,194,17,0.273,'prediccion de regularmente'),(166,195,17,0.273,'prediccion de bueno'),(167,195,17,0.091,'prediccion de malo'),(168,195,17,0.182,'prediccion de muy bueno'),(169,195,17,0.364,'prediccion de no pase'),(170,195,17,0.091,'prediccion de regularmente'),(171,197,17,0.182,'prediccion de bueno'),(172,197,17,0.182,'prediccion de malo'),(173,197,17,0.273,'prediccion de muy bueno'),(174,197,17,0.091,'prediccion de no pase'),(175,197,17,0.273,'prediccion de regularmente'),(176,198,17,0.273,'prediccion de bueno'),(177,198,17,0.182,'prediccion de malo'),(178,198,17,0.091,'prediccion de muy bueno'),(179,198,17,0.273,'prediccion de no pase'),(180,198,17,0.182,'prediccion de regularmente'),(181,201,17,0.182,'prediccion de bueno'),(182,201,17,0.273,'prediccion de malo'),(183,201,17,0.091,'prediccion de muy bueno'),(184,201,17,0.182,'prediccion de no pase'),(185,201,17,0.273,'prediccion de regularmente'),(186,199,17,0.091,'prediccion de bueno'),(187,199,17,0.091,'prediccion de malo'),(188,199,17,0.182,'prediccion de muy bueno'),(189,199,17,0.455,'prediccion de no pase'),(190,199,17,0.182,'prediccion de regularmente'),(191,204,17,0.273,'prediccion de bueno'),(192,204,17,0.091,'prediccion de malo'),(193,204,17,0.182,'prediccion de muy bueno'),(194,204,17,0.182,'prediccion de no pase'),(195,204,17,0.273,'prediccion de regularmente'),(196,207,17,0.273,'prediccion de bueno'),(197,207,17,0.182,'prediccion de malo'),(198,207,17,0.091,'prediccion de muy bueno'),(199,207,17,0.273,'prediccion de no pase'),(200,207,17,0.182,'prediccion de regularmente'),(201,206,17,0.364,'prediccion de bueno'),(202,206,17,0.182,'prediccion de malo'),(203,206,17,0.273,'prediccion de muy bueno'),(204,206,17,0.091,'prediccion de no pase'),(205,206,17,0.091,'prediccion de regularmente'),(206,208,18,0.364,'prediccion de bueno'),(207,208,18,0.273,'prediccion de malo'),(208,208,18,0.091,'prediccion de muy bueno'),(209,208,18,0.091,'prediccion de no pase'),(210,208,18,0.182,'prediccion de regularmente'),(211,211,18,0.364,'prediccion de bueno'),(212,211,18,0.091,'prediccion de malo'),(213,211,18,0.182,'prediccion de muy bueno'),(214,211,18,0.182,'prediccion de no pase'),(215,211,18,0.182,'prediccion de regularmente'),(216,210,18,0.091,'prediccion de bueno'),(217,210,18,0.091,'prediccion de malo'),(218,210,18,0.182,'prediccion de muy bueno'),(219,210,18,0.455,'prediccion de no pase'),(220,210,18,0.182,'prediccion de regularmente'),(221,212,18,0.455,'prediccion de bueno'),(222,212,18,0.273,'prediccion de malo'),(223,212,18,0.091,'prediccion de muy bueno'),(224,212,18,0.091,'prediccion de no pase'),(225,212,18,0.091,'prediccion de regularmente'),(226,214,18,0.273,'prediccion de bueno'),(227,214,18,0.091,'prediccion de malo'),(228,214,18,0.273,'prediccion de muy bueno'),(229,214,18,0.182,'prediccion de no pase'),(230,214,18,0.182,'prediccion de regularmente'),(231,217,18,0.455,'prediccion de bueno'),(232,217,18,0.091,'prediccion de malo'),(233,217,18,0.182,'prediccion de muy bueno'),(234,217,18,0.182,'prediccion de no pase'),(235,217,18,0.091,'prediccion de regularmente'),(236,223,18,0.182,'prediccion de bueno'),(237,223,18,0.182,'prediccion de malo'),(238,223,18,0.182,'prediccion de muy bueno'),(239,223,18,0.182,'prediccion de no pase'),(240,223,18,0.273,'prediccion de regularmente'),(241,227,18,0.182,'prediccion de bueno'),(242,227,18,0.182,'prediccion de malo'),(243,227,18,0.273,'prediccion de muy bueno'),(244,227,18,0.182,'prediccion de no pase'),(245,227,18,0.182,'prediccion de regularmente'),(246,229,18,0.091,'prediccion de bueno'),(247,229,18,0.182,'prediccion de malo'),(248,229,18,0.273,'prediccion de muy bueno'),(249,229,18,0.091,'prediccion de no pase'),(250,229,18,0.364,'prediccion de regularmente'),(251,230,23,0.182,'prediccion de bueno'),(252,230,23,0.182,'prediccion de malo'),(253,230,23,0.273,'prediccion de muy bueno'),(254,230,23,0.091,'prediccion de no pase'),(255,230,23,0.273,'prediccion de regularmente');
/*!40000 ALTER TABLE `Prediction` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Questions`
--

DROP TABLE IF EXISTS `Questions`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Questions` (
  `question_id` int NOT NULL AUTO_INCREMENT,
  `exam_id` int DEFAULT NULL,
  `question_text` text NOT NULL,
  `image_path` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`question_id`),
  KEY `exam_id` (`exam_id`),
  CONSTRAINT `Questions_ibfk_1` FOREIGN KEY (`exam_id`) REFERENCES `Exams` (`exam_id`)
) ENGINE=InnoDB AUTO_INCREMENT=46 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Questions`
--

LOCK TABLES `Questions` WRITE;
/*!40000 ALTER TABLE `Questions` DISABLE KEYS */;
INSERT INTO `Questions` VALUES (5,3,'¿Qué es la programación?',NULL),(6,3,'¿Cuál es el lenguaje de programación más usado?',NULL),(25,22,'¿Qué es la programación?s',NULL),(26,22,'¿Cuál es el lenguaje de programación más usado?',NULL),(27,22,'¿que es la predicion?',NULL),(28,23,'¿Cuál es el valor de x en la ecuación \n2x+3=7?',NULL),(29,23,'¿Cuál es el valor de x=3?',NULL),(30,23,'Simplifica la expresión 4(2x+1)',NULL),(31,23,'¿Cuál es la propiedad que indica a+b=b+a?',NULL),(32,23,'¿Cuál es el valor de x en la ecuación 3x−5=1?',NULL),(33,24,'¿Cuál es el valor de ? en la ecuación 4? − 7 = 9?',NULL),(34,24,'Resuelve la desigualdad 2x+3<11',NULL),(35,24,'¿Cuál es el valor de ? en la ecuación 6x+2=14?',NULL),(36,25,'PRENT',NULL),(37,56,'Juan tiene 4 canicas. Luis tiene el doble de canicas que Juan.\n¿Cuántas canicas tiene Luis?',NULL),(38,56,'Los panes chutas son originarios del Cusco. Estos se\ncaracterizan por ser panes muy grandes de forma circular.\nJuana quiere repartir 4 de estos panes entre sus 3 sobrinos,\nde tal forma que cada uno reciba la misma cantidad de pan.\nAproximadamente, ¿qué cantidad le corresponde a cada\nuno de sus sobrinos?',NULL),(39,56,'Carlos compró cierta cantidad de panes. Puso 1/4 de esa cantidad sobre una bandeja y dejó el resto de panes en la bolsa',NULL),(40,56,'Pedro tiene 15*3/4 kg de azúcar en su tienda. Con esta cantidad de azúcar, él armará bolsas de 1/2 kg de azúcar cada una.\n¿Cuántas bolsas cómo máximo pudo armar Pedro? ',NULL),(41,56,'Un grupo de 48 personas se presentó a diversos puestos de trabajo ofrecidos por una empresa. La cantidad de mujeres que se presentaron a estos puestos de trabajo triplicó a la cantidad de varones.\nSegún esta información, ¿cuántas mujeres se presentaron a estos puestos de trabajo?',NULL),(42,56,'Resuelve la siguiente ecuación:\n4 (x + 1) = 28 ¿Cuál es el valor de “x”?',NULL),(43,57,'',NULL),(44,58,'PRENGUNTA 1 ',NULL),(45,58,'PREGUNTA 2',NULL);
/*!40000 ALTER TABLE `Questions` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Students`
--

DROP TABLE IF EXISTS `Students`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Students` (
  `student_id` int NOT NULL AUTO_INCREMENT,
  `user_id` int DEFAULT NULL,
  `first_name` varchar(50) DEFAULT NULL,
  `last_name` varchar(50) DEFAULT NULL,
  `birthdate` date DEFAULT NULL,
  `phone` varchar(15) DEFAULT NULL,
  `district` varchar(100) DEFAULT NULL,
  `gender` varchar(50) DEFAULT NULL,
  `economic_level` varchar(50) DEFAULT NULL,
  `access_platform` varchar(50) DEFAULT NULL,
  `grade_id` int DEFAULT NULL,
  PRIMARY KEY (`student_id`),
  KEY `user_id` (`user_id`),
  KEY `grade_id` (`grade_id`),
  CONSTRAINT `Students_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `Users` (`user_id`),
  CONSTRAINT `Students_ibfk_2` FOREIGN KEY (`grade_id`) REFERENCES `Grades` (`grade_id`)
) ENGINE=InnoDB AUTO_INCREMENT=232 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Students`
--

LOCK TABLES `Students` WRITE;
/*!40000 ALTER TABLE `Students` DISABLE KEYS */;
INSERT INTO `Students` VALUES (158,29,'María','García','2005-04-20','999333444','San Isidro','Femenino','Alto','1h - 4h',1),(159,30,'Pedro','Martínez','2005-02-10','999555666','San Borja','Masculino','Bajo','1h - 4h',1),(160,31,'Luisa','López','2005-01-25','999777888','Surco','Femenino','Medio','1h - 4h',1),(161,32,'Jorge','Hernández','2005-03-05','999999000','San Miguel','Masculino','Medio','1h - 4h',1),(162,33,'Ana','Rodríguez','2005-04-12','999111222','Jesus María','Femenino','Alto','1h - 4h',1),(163,34,'Carlos','Gómez','2005-02-18','999333444','Magdalena','Masculino','Bajo','1h - 4h',1),(164,35,'Laura','Torres','2005-01-30','999555666','Pueblo Libre','Femenino','Medio','1h - 4h',1),(165,36,'Gabriel','Sánchez','2005-03-20','999777888','San Juan de Lurigancho','Masculino','Bajo','1h - 4h',1),(166,37,'Sofía','Paz','2005-04-08','999999000','Chorrillos','Femenino','Alto','1h - 4h',1),(167,38,'Diego','Ramírez','2005-02-22','999111222','Barranco','Masculino','Medio','1h - 4h',1),(168,39,'Valentina','Mendoza','2005-01-12','999333444','La Molina','Femenino','Medio','1h - 4h',1),(169,40,'Mateo','Guerrero','2005-03-25','999555666','Lince','Masculino','Alto','1h - 4h',1),(170,41,'Camila','Flores','2005-04-05','999777888','San Isidro','Femenino','Bajo','1h - 4h',1),(171,42,'Lucas','Díaz','2005-02-28','999999000','Surquillo','Masculino','Alto','1h - 4h',1),(172,43,'Martín','Castillo','2004-03-17','999111222','Miraflores','Masculino','Medio','1h - 4h',2),(173,44,'Renata','Vargas','2004-04-21','999333444','San Isidro','Femenino','Alto','1h - 4h',2),(174,45,'Fernando','Cabrera','2004-02-11','999555666','San Borja','Masculino','Bajo','1h - 4h',2),(175,46,'Marcela','Montoya','2004-01-26','999777888','Surco','Femenino','Medio','1h - 4h',2),(176,47,'Simón','Ríos','2004-03-07','999999000','San Miguel','Masculino','Medio','1h - 4h',2),(177,48,'Valeria','Lara','2004-04-14','999111222','Jesus María','Femenino','Alto','1h - 4h',2),(178,49,'Eduardo','Delgado','2004-02-19','999333444','Magdalena','Masculino','Bajo','1h - 4h',2),(179,50,'Abril','Santana','2004-01-31','999555666','Pueblo Libre','Femenino','Medio','1h - 4h',2),(180,51,'Felipe','Ortega','2004-03-22','999777888','San Juan de Lurigancho','Masculino','Bajo','1h - 4h',2),(181,52,'Antonella','Peralta','2004-04-09','999999000','Chorrillos','Femenino','Alto','1h - 4h',2),(182,53,'Rodrigo','Navarro','2004-02-23','999111222','Barranco','Masculino','Medio','1h - 4h',2),(183,54,'Julieta','Herrera','2004-01-13','999333444','La Molina','Femenino','Medio','1h - 4h',2),(184,55,'Luciana','Silva','2004-03-27','999555666','Lince','Femenino','Alto','1h - 4h',2),(185,56,'Sebastián','Moreno','2004-04-06','999777888','San Isidro','Masculino','Bajo','1h - 4h',2),(186,57,'Valentina','Gutiérrez','2004-02-29','999999000','Surquillo','Femenino','Alto','1h - 4h',2),(187,58,'Emilio','Vega','2003-03-19','999111222','Miraflores','Masculino','Medio','1h - 4h',3),(188,59,'Carolina','Reyes','2003-04-22','999333444','San Isidro','Femenino','Alto','1h - 4h',3),(189,60,'Javier','Mendoza','2003-02-12','999555666','San Borja','Masculino','Bajo','1h - 4h',3),(190,61,'Daniela','Paz','2003-01-27','999777888','Surco','Femenino','Medio','1h - 4h',3),(191,62,'Ricardo','Fernández','2003-03-08','999999000','San Miguel','Masculino','Medio','1h - 4h',3),(192,63,'Laura','Guerra','2003-04-15','999111222','Jesus María','Femenino','Alto','1h - 4h',3),(193,64,'Roberto','Soto','2003-02-20','999333444','Magdalena','Masculino','Bajo','1h - 4h',3),(194,65,'Luciana','Castañeda','2003-01-01','999555666','Pueblo Libre','Femenino','Medio','1h - 4h',3),(195,66,'Diego','Herrera','2003-03-23','999777888','San Juan de Lurigancho','Masculino','Bajo','1h - 4h',3),(196,67,'Valeria','Sánchez','2003-04-10','999999000','Chorrillos','Femenino','Alto','1h - 4h',3),(197,68,'Martín','Ortiz','2003-02-25','999111222','Barranco','Masculino','Medio','1h - 4h',3),(198,69,'Antonella','Castro','2003-01-15','999333444','La Molina','Femenino','Medio','1h - 4h',3),(199,70,'Lucas','Gómez','2003-03-29','999555666','Lince','Masculino','Alto','1h - 4h',3),(200,71,'Camila','Flores','2003-04-07','999777888','San Isidro','Femenino','Bajo','1h - 4h',3),(201,72,'Santiago','Vidal','2003-02-28','999999000','Surquillo','Masculino','Alto','1h - 4h',3),(202,73,'Isabella','Pérez','2002-03-18','999111222','Miraflores','Femenino','Alto','1h - 4h',4),(203,74,'Matías','García','2002-04-23','999333444','San Isidro','Masculino','Medio','1h - 4h',4),(204,75,'Valentina','Martínez','2002-02-13','999555666','San Borja','Femenino','Bajo','1h - 4h',4),(205,76,'Luciano','López','2002-01-28','999777888','Surco','Masculino','Medio','1h - 4h',4),(206,77,'Julieta','Hernández','2002-03-09','999999000','San Miguel','Femenino','Alto','1h - 4h',4),(207,78,'Emiliano','Rodríguez','2002-04-16','999111222','Jesus María','Masculino','Medio','1h - 4h',4),(208,79,'Isidora','Gómez','2002-02-21','999333444','Magdalena','Femenino','Bajo','1h - 4h',4),(209,80,'Sebastián','Delgado','2002-01-02','999555666','Pueblo Libre','Masculino','Medio','1h - 4h',4),(210,81,'Valeria','Santos','2002-03-24','999777888','San Juan de Lurigancho','Femenino','Bajo','1h - 4h',4),(211,82,'Tomás','Ortega','2002-04-08','999999000','Chorrillos','Masculino','Alto','1h - 4h',4),(212,83,'Mariana','Peralta','2002-02-23','999111222','Barranco','Femenino','Medio','1h - 4h',4),(213,84,'Gabriel','Silva','2002-01-14','999333444','La Molina','Masculino','Medio','1h - 4h',4),(214,85,'Carolina','Flores','2002-03-28','999555666','Lince','Femenino','Alto','1h - 4h',4),(215,86,'Emilia','Pérez','2001-03-18','999111222','Miraflores','Femenino','Alto','1h - 4h',5),(216,87,'Felipe','García','2001-04-23','999333444','San Isidro','Masculino','Medio','1h - 4h',5),(217,88,'Valentina','Martínez','2001-02-13','999555666','San Borja','Femenino','Bajo','1h - 4h',5),(218,89,'Lucas','López','2001-01-28','999777888','Surco','Masculino','Medio','1h - 4h',5),(219,90,'Julieta','Hernández','2001-03-09','999999000','San Miguel','Femenino','Alto','1h - 4h',5),(220,91,'Emiliano','Rodríguez','2001-04-16','999111222','Jesus María','Masculino','Medio','1h - 4h',5),(221,92,'Isidora','Gómez','2001-02-21','999333444','Magdalena','Femenino','Bajo','1h - 4h',5),(222,93,'Sebastián','Delgado','2001-01-02','999555666','Pueblo Libre','Masculino','Medio','1h - 4h',5),(223,94,'Valeria','Santos','2001-03-24','999777888','San Juan de Lurigancho','Femenino','Bajo','1h - 4h',5),(224,95,'Tomás','Ortega','2001-04-08','999999000','Chorrillos','Masculino','Alto','1h - 4h',5),(225,96,'Mariana','Peralta','2001-02-23','999111222','Barranco','Femenino','Medio','1h - 4h',5),(226,97,'Gabriel','Silva','2001-01-14','999333444','La Molina','Masculino','Medio','1h - 4h',5),(227,98,'Carolina','Flores','2001-03-28','999555666','Lince','Femenino','Alto','1h - 4h',5),(228,99,'Emilio','Vargas','2001-04-06','999777888','San Isidro','Masculino','Bajo','1h - 4h',5),(229,100,'Martina','Moreno','2001-02-20','999999000','Surquillo','Femenino','Alto','1h - 4h',5),(230,104,'Piere Prueba','Prueba Edicion','2000-10-10','999888777','Lima','Masculino','Medio','1h - 2h',2),(231,108,'aprueba gjhgj','rueba hjghj','2023-11-02','65865878','Chorrillos','Femenino','Medio','2h - 4h',2);
/*!40000 ALTER TABLE `Students` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Teachers`
--

DROP TABLE IF EXISTS `Teachers`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Teachers` (
  `teacher_id` int NOT NULL AUTO_INCREMENT,
  `user_id` int DEFAULT NULL,
  `first_name` varchar(50) DEFAULT NULL,
  `last_name` varchar(50) DEFAULT NULL,
  `birthdate` date DEFAULT NULL,
  `phone` varchar(15) DEFAULT NULL,
  `district` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`teacher_id`),
  KEY `user_id` (`user_id`),
  CONSTRAINT `Teachers_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `Users` (`user_id`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Teachers`
--

LOCK TABLES `Teachers` WRITE;
/*!40000 ALTER TABLE `Teachers` DISABLE KEYS */;
INSERT INTO `Teachers` VALUES (1,3,'joel','san','1985-10-25','1234567890','Lima'),(2,1,'Joel1','Ch','2000-02-22','854574547','Lima');
/*!40000 ALTER TABLE `Teachers` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Units`
--

DROP TABLE IF EXISTS `Units`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Units` (
  `unit_id` int NOT NULL AUTO_INCREMENT,
  `course_id` int DEFAULT NULL,
  `unit_name` varchar(100) NOT NULL,
  PRIMARY KEY (`unit_id`),
  KEY `course_id` (`course_id`),
  CONSTRAINT `Units_ibfk_1` FOREIGN KEY (`course_id`) REFERENCES `Courses` (`course_id`)
) ENGINE=InnoDB AUTO_INCREMENT=54 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Units`
--

LOCK TABLES `Units` WRITE;
/*!40000 ALTER TABLE `Units` DISABLE KEYS */;
INSERT INTO `Units` VALUES (1,7,'Unidad 1: Introducción'),(2,6,'Unidad 1: Introducción'),(3,6,'Unidad 2: Desarrollo Metodologias'),(4,6,'Unidad 3: Conceptos Logicos'),(5,14,'Unidad 1: inicio '),(6,14,'Unidad 2: desarrollo '),(7,5,'Unidad 1'),(8,15,'Unidad 1'),(9,15,'Unidad 2'),(10,15,'Unidad 3'),(11,16,'Unidad 1: Fundamentos de Álgebra'),(12,16,'Unidad 2: Ecuaciones y Desigualdades Lineales'),(13,5,'hgfhg'),(19,16,'Unidad 1 - Álgebra básica'),(20,16,'Unidad 2 - Geometría elemental'),(21,16,'Unidad 3 - Problemas de aplicación'),(22,16,'Unidad 4 - Ecuaciones lineales'),(23,16,'Unidad 5 - Geometría avanzada'),(24,17,'Unidad 1 - Gramática básica'),(25,17,'Unidad 2 - Vocabulario esencial'),(26,17,'Unidad 3 - Comprensión auditiva'),(27,17,'Unidad 4 - Comprensión de lectura'),(28,17,'Unidad 5 - Conversación básica'),(29,18,'Unidad 1 - Introducción a la biología'),(30,18,'Unidad 2 - Fundamentos de química'),(31,18,'Unidad 3 - Conceptos básicos de física'),(32,18,'Unidad 4 - Experimentación científica'),(33,18,'Unidad 5 - Teoría atómica y molecular'),(34,19,'Unidad 1 - Historia local'),(35,19,'Unidad 2 - Culturas antiguas'),(36,19,'Unidad 3 - Edad Media y Renacimiento'),(37,19,'Unidad 4 - Revoluciones e independencias'),(38,19,'Unidad 5 - Siglo XX: Guerras y globalización'),(39,20,'Unidad 1 - Introducción al deporte'),(40,20,'Unidad 2 - Fundamentos del ejercicio físico'),(41,20,'Unidad 3 - Técnicas deportivas'),(42,20,'Unidad 4 - Salud y bienestar'),(43,20,'Unidad 5 - Competencias y habilidades físicas'),(44,21,'Unidad 1 - Introducción al arte'),(45,21,'Unidad 2 - Técnicas artísticas básicas'),(46,21,'Unidad 3 - Estilos y corrientes artísticas'),(47,21,'Unidad 4 - Expresión creativa'),(48,21,'Unidad 5 - Proyecto final: Creación artística'),(49,23,'UNIDAD 1'),(50,23,'UNIDAD 2'),(51,23,'UNIDAD 3'),(52,23,'UNIDAD 4'),(53,48,'UNIDAD 1');
/*!40000 ALTER TABLE `Units` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Users`
--

DROP TABLE IF EXISTS `Users`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Users` (
  `user_id` int NOT NULL AUTO_INCREMENT,
  `username` varchar(50) NOT NULL,
  `password` varchar(255) NOT NULL,
  `email` varchar(100) NOT NULL,
  `role` enum('student','admin') NOT NULL,
  PRIMARY KEY (`user_id`),
  UNIQUE KEY `username` (`username`),
  UNIQUE KEY `email` (`email`)
) ENGINE=InnoDB AUTO_INCREMENT=109 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Users`
--

LOCK TABLES `Users` WRITE;
/*!40000 ALTER TABLE `Users` DISABLE KEYS */;
INSERT INTO `Users` VALUES (1,'joel1','123','joel@example.com','admin'),(2,'piere1','123','piere@example.com','student'),(3,'joel','123','j30@example.com','admin'),(29,'student1','password1','student1@example.com','student'),(30,'student2','password2','student2@example.com','student'),(31,'student3','password3','student3@example.com','student'),(32,'student4','password4','student4@example.com','student'),(33,'student5','password5','student5@example.com','student'),(34,'student6','password6','student6@example.com','student'),(35,'student7','password7','student7@example.com','student'),(36,'student8','password8','student8@example.com','student'),(37,'student9','password9','student9@example.com','student'),(38,'student10','password10','student10@example.com','student'),(39,'student11','password11','student11@example.com','student'),(40,'student12','password12','student12@example.com','student'),(41,'student13','password13','student13@example.com','student'),(42,'student14','password14','student14@example.com','student'),(43,'student15','password15','student15@example.com','student'),(44,'student16','password16','student16@example.com','student'),(45,'student17','password17','student17@example.com','student'),(46,'student18','password18','student18@example.com','student'),(47,'student19','password19','student19@example.com','student'),(48,'student20','password20','student20@example.com','student'),(49,'student21','password21','student21@example.com','student'),(50,'student22','password22','student22@example.com','student'),(51,'student23','password23','student23@example.com','student'),(52,'student24','password24','student24@example.com','student'),(53,'student25','password25','student25@example.com','student'),(54,'student26','password26','student26@example.com','student'),(55,'student27','password27','student27@example.com','student'),(56,'student28','password28','student28@example.com','student'),(57,'student29','password29','student29@example.com','student'),(58,'student30','password30','student30@example.com','student'),(59,'student31','password31','student31@example.com','student'),(60,'student32','password32','student32@example.com','student'),(61,'student33','password33','student33@example.com','student'),(62,'student34','password34','student34@example.com','student'),(63,'student35','password35','student35@example.com','student'),(64,'student36','password36','student36@example.com','student'),(65,'student37','password37','student37@example.com','student'),(66,'student38','password38','student38@example.com','student'),(67,'student39','password39','student39@example.com','student'),(68,'student40','password40','student40@example.com','student'),(69,'student41','password41','student41@example.com','student'),(70,'student42','password42','student42@example.com','student'),(71,'student43','password43','student43@example.com','student'),(72,'student44','password44','student44@example.com','student'),(73,'student45','password45','student45@example.com','student'),(74,'student46','password46','student46@example.com','student'),(75,'student47','password47','student47@example.com','student'),(76,'student48','password48','student48@example.com','student'),(77,'student49','password49','student49@example.com','student'),(78,'student50','password50','student50@example.com','student'),(79,'student51','password51','student51@example.com','student'),(80,'student52','password52','student52@example.com','student'),(81,'student53','password53','student53@example.com','student'),(82,'student54','password54','student54@example.com','student'),(83,'student55','password55','student55@example.com','student'),(84,'student56','password56','student56@example.com','student'),(85,'student57','password57','student57@example.com','student'),(86,'student58','password58','student58@example.com','student'),(87,'student59','password59','student59@example.com','student'),(88,'student60','password60','student60@example.com','student'),(89,'student61','password61','student61@example.com','student'),(90,'student62','password62','student62@example.com','student'),(91,'student63','password63','student63@example.com','student'),(92,'student64','password64','student64@example.com','student'),(93,'student65','password65','student65@example.com','student'),(94,'student66','password66','student66@example.com','student'),(95,'student67','password67','student67@example.com','student'),(96,'student68','password68','student68@example.com','student'),(97,'student69','password69','student69@example.com','student'),(98,'student70','password70','student70@example.com','student'),(99,'student71','password71','student71@example.com','student'),(100,'student72','password72','student72@example.com','student'),(101,'student73','password73','student73@example.com','student'),(102,'student74','password74','student74@example.com','student'),(103,'student75','password75','student75@example.com','student'),(104,'piere','1234','piere@gmail.com','student'),(108,'prueba','prueba12345','prueba@gmail.com','student');
/*!40000 ALTER TABLE `Users` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2024-10-19  6:19:32
